"""Cooperative, cross-process lock for a shared microscopy GPU."""

from __future__ import annotations

import errno
import getpass
import json
import os
import socket
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

if os.name == "nt":
    import msvcrt
else:
    import fcntl


@dataclass
class _HeldLock:
    file: BinaryIO
    count: int = 1


_held: dict[Path, _HeldLock] = {}
_guard = threading.Lock()


def _after_fork_child() -> None:
    # A forked child inherits the parent's descriptors and must close its
    # copies without explicitly unlocking the parent's file description.
    for held in _held.values():
        held.file.close()
    _held.clear()
    _guard.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_guard.acquire,
        after_in_parent=_guard.release,
        after_in_child=_after_fork_child,
    )


def _temp_lock_path() -> Path:
    return Path(tempfile.gettempdir()) / "lab-microscopy-gpu.lock"


def default_gpu_lock_path(lock_path: str | Path | None = None) -> Path:
    """Resolve the configured path before testing whether it can be opened."""
    if lock_path is not None:
        return Path(lock_path)
    configured = os.environ.get("LAB_MICROSCOPY_GPU_LOCK")
    if configured:
        return Path(configured)
    if os.name == "nt":
        return Path(os.environ.get("PROGRAMDATA", tempfile.gettempdir())) / "lab-microscopy" / "gpu.lock"
    return _temp_lock_path()


def gpu_lock_disabled() -> bool:
    return os.environ.get("LAB_MICROSCOPY_DISABLE_GPU_LOCK", "").strip().lower() in ("1", "true", "yes")


def _open_lock_file(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Both OS locks work on a read-only descriptor. Lab accounts can share
    # an existing file even when only the creator has write permission.
    created = False
    try:
        file = path.open("rb")
    except FileNotFoundError:
        try:
            file = path.open("x+b")
            created = True
        except FileExistsError:
            file = path.open("rb")

    try:
        if created and os.name != "nt":
            # A private umask must not prevent other lab accounts from locking
            # the same file. Existing file permissions are left to its owner.
            os.fchmod(file.fileno(), 0o644)
        if os.name == "nt" and os.fstat(file.fileno()).st_size == 0:
            try:
                with path.open("r+b") as writer:
                    writer.write(b"0")
            except OSError:
                # Another process may have initialized and locked the byte
                # between our size check and write.
                if os.fstat(file.fileno()).st_size == 0:
                    raise
        return file
    except BaseException:
        file.close()
        raise


def _try_lock(file: BinaryIO) -> bool:
    try:
        if os.name == "nt":
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            return False
        raise
    return True


def _unlock(file: BinaryIO) -> None:
    try:
        with suppress(OSError):
            if os.name == "nt":
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    finally:
        file.close()


def _write_holder(path: Path, job: str) -> None:
    temporary = None
    try:
        holder = {
            "pid": os.getpid(),
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "job": job,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=path.name + ".", dir=path.parent, delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(holder, file)
            if os.name != "nt":
                os.fchmod(file.fileno(), 0o644)
        os.replace(temporary, path)
    except (OSError, KeyError):
        # Reporting ownership is optional, including in a shared directory
        # where a lab account may lack permission to replace metadata.
        pass
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _holder_details(path: Path) -> str:
    try:
        holder = json.loads(path.read_text(encoding="utf-8"))
        return (
            f"PID {holder['pid']} ({holder['user']}@{holder['host']}), "
            f"job {holder['job']!r}, since {holder['started_at']}"
        )
    except (OSError, ValueError, KeyError, TypeError):
        return "holder details unavailable"


@contextmanager
def gpu_lock(
    lock_path: str | Path | None = None,
    timeout_s: float | None = None,
    poll_interval_s: float = 1.0,
    job: str = "",
    notify: Callable[[str], None] | None = None,
) -> Iterator[None]:
    """Wait for exclusive GPU access, sharing nested acquisitions in this process.

    The lock file stays in place so every contender opens the same file.
    Closing the descriptor or exiting the process releases the OS lock.
    """
    if gpu_lock_disabled():
        yield
        return

    path = default_gpu_lock_path(lock_path)
    fallback = None
    if os.name == "nt" and lock_path is None and not os.environ.get("LAB_MICROSCOPY_GPU_LOCK"):
        fallback = _temp_lock_path()

    path = path.resolve()
    holder_path = path.with_name(path.name + ".holder.json")
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    next_notice = time.monotonic()
    owner_pid = os.getpid()

    while True:
        with _guard:
            held = _held.get(path)
            if held is not None:
                held.count += 1
                acquired = False
                break
            try:
                file = _open_lock_file(path)
            except OSError:
                if fallback is None:
                    raise
                path = fallback.resolve()
                holder_path = path.with_name(path.name + ".holder.json")
                fallback = None
                continue
            try:
                acquired = _try_lock(file)
                if acquired:
                    _held[path] = _HeldLock(file)
                    break
            finally:
                if path not in _held:
                    file.close()

        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise TimeoutError(f"Timed out waiting for GPU lock {path} ({_holder_details(holder_path)})")
        if now >= next_notice:
            message = f"Waiting for GPU lock {path} ({_holder_details(holder_path)})"
            if notify is None:
                print(message, file=sys.stderr, flush=True)
            else:
                notify(message)
            next_notice = now + 30.0

        if deadline is None:
            time.sleep(poll_interval_s)
        else:
            time.sleep(max(0.0, min(poll_interval_s, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for GPU lock {path} ({_holder_details(holder_path)})")

    try:
        if acquired:
            _write_holder(holder_path, job)
        yield
    finally:
        if os.getpid() == owner_pid:
            with _guard:
                held = _held[path]
                held.count -= 1
                if held.count == 0:
                    try:
                        with suppress(OSError):
                            holder_path.unlink(missing_ok=True)
                    finally:
                        del _held[path]
                        _unlock(held.file)
