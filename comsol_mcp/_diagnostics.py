"""Argument-free stderr/file diagnostics that do not affect connection behavior."""
from contextlib import contextmanager
from datetime import datetime, timezone
import functools
import inspect
import os
from pathlib import Path
import sys
import threading
import time


_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "mcp-diagnostics.log"
_emit_lock = threading.RLock()


def _emit(stage, event, elapsed=None):
    try:
        stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        duration = "" if elapsed is None else f" elapsed_seconds={elapsed:.6f}"
        # Keep diagnostics limited to stage metadata, never call arguments,
        # return values or exceptions (which may contain credentials).
        record = (f"{stamp} pid={os.getpid()} thread_id={threading.get_ident()} "
                  f"stage={stage} event={event}{duration}")
        with _emit_lock:
            try:
                print(record, file=sys.stderr, flush=True)
            except Exception:
                pass
            try:
                _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with _LOG_PATH.open("a", encoding="utf-8") as log:
                    log.write(record + "\n")
                    log.flush()
            except Exception:
                pass
    except Exception:
        pass  # Logging failures must not change connection behavior.


@contextmanager
def phase(stage):
    started = time.monotonic()
    _emit(stage, "start")
    try:
        yield
    finally:
        _emit(stage, "end", time.monotonic() - started)


def traced(stage):
    def decorate(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def invoke(*args, **kwargs):
                with phase(stage):
                    return await fn(*args, **kwargs)
        else:
            @functools.wraps(fn)
            def invoke(*args, **kwargs):
                with phase(stage):
                    return fn(*args, **kwargs)
        return invoke
    return decorate
