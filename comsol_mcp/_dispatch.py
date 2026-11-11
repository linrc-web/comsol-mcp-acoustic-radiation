"""Async MCP boundary and one daemon worker for synchronous tool execution.

Timeouts abandon the wait, never the running Java call. The job remains owned
by the worker until it actually returns. No MPH import occurs in this module.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import comsol_mcp._server as _srv
from comsol_mcp._diagnostics import phase, traced


@dataclass
class _Job:
    name: str
    callback: Any
    deadline: float | None
    started: float = field(default_factory=time.monotonic)
    future: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)
    stage: str = "queued"
    expired: bool = False


class WorkerBusy(RuntimeError):
    def __init__(self, job):
        super().__init__("A connection operation is still pending; no overlapping operation was started.")
        self.job = job


class ToolWorker:
    def __init__(self):
        self.lock = threading.RLock()
        self.jobs = queue.Queue()
        self.active: _Job | None = None
        self.pending: list[_Job] = []
        self.thread = threading.Thread(target=self._loop, name="comsol-mph-worker", daemon=True)
        with phase("worker_start"):
            self.thread.start()

    def submit(self, name, callback, timeout=None):
        with self.lock:
            for job in self.pending:
                if job.expired or job.name in {"server_connect", "server_start", "server_disconnect"}:
                    raise WorkerBusy(job)
            job = _Job(name, callback, None if timeout is None else time.monotonic() + timeout)
            self.pending.append(job)
            self.jobs.put(job)
            return job

    def _loop(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            with self.lock:
                self.active = job
            try:
                checkpoint("dispatch")
                with phase("worker_call"):
                    result = job.callback()
                checkpoint("completed")
                error = None
            except BaseException as exc:
                error = exc
            finally:
                with self.lock:
                    if job.expired:
                        _srv._client_connected = False
                        _srv._client_connection_uncertain = _srv._client is not None
                    self.pending.remove(job)
                    self.active = None
                if error is None:
                    job.future.set_result(result)
                else:
                    job.future.set_exception(error)
                self.jobs.task_done()

    def close(self):
        """For clean shutdown once all calls have returned; never kill Java."""
        self.jobs.put(None)
        self.thread.join(timeout=2)

    def expire(self, job):
        with self.lock:
            job.expired = True
            return job.stage


def current_job():
    runtime = _srv._connection_worker
    if runtime is not None and threading.current_thread() is runtime.thread:
        return runtime.active
    return None


def checkpoint(stage: str):
    job = current_job()
    if job is None:
        return
    runtime = _srv._connection_worker
    with runtime.lock:
        if job.expired or (job.deadline is not None and time.monotonic() >= job.deadline):
            job.expired = True
            raise TimeoutError(f"Total timeout in stage {job.stage}; the underlying call was not forcibly terminated.")
        job.stage = stage
    logging.getLogger(__name__).info("Tool %s stage=%s elapsed=%.3fs", job.name, stage, time.monotonic() - job.started)


def remaining(default: float) -> float:
    job = current_job()
    if job is None or job.deadline is None:
        return default
    return max(0.0, min(default, job.deadline - time.monotonic()))


def failure_result(name, error, *, stage="dispatch", job=None):
    # Deliberately no runtime lock, filesystem IO, or model/server Java access.
    from comsol_mcp._state import _json, _now_iso
    return _json({
        "success": False, "tool": name, "timestamp": time.time(), "datetime": _now_iso(),
        "server": {"version": _srv.VERSION, "timestamp": time.time(), "datetime": _now_iso(),
                   "status": "unconfirmed", "server_running": False,
                   "server_started_by_mcp": _srv._server_started_by_mcp,
                   "attached_to_existing_server": False,
                   "server_host": _srv._connected_host or _srv.DEFAULT_HOST,
                   "server_port": _srv._connected_port, "server_endpoint": "",
                   "desktop_client_role": "visual-client", "comsol_root": str(_srv.COMSOL_ROOT),
                   "mcp_home": str(_srv.COMSOL_SERVER_MCP_HOME), "logs_dir": str(_srv.LOGS_DIR),
                   "outputs_dir": str(_srv.OUTPUTS_DIR), "current_model_label": "",
                   "current_model_path": _srv._current_model_path,
                   "current_model_origin": _srv._current_model_origin,
                   "last_command": name, "last_error": str(error),
                   "recommended_desktop_flow": _srv.RECOMMENDED_DESKTOP_FLOW,
                   "desktop_should_connect": {"host": _srv._connected_host or _srv.DEFAULT_HOST,
                                               "port": _srv._connected_port},
                   "workflow": {}, "state_confirmed": False},
        "data": {"stage": stage, "elapsed_seconds": 0 if job is None else time.monotonic() - job.started,
                 "operation_pending": job is not None and not job.future.done()},
        "error": str(error), "log_path": str(_srv.SERVER_LOG),
        "operations_path": str(_srv.OPERATIONS_FILE),
    })


def _budget(fn, args, kwargs):
    if fn.__name__ not in {"server_connect", "server_start", "server_disconnect"}:
        return None
    values = inspect.signature(fn).bind(*args, **kwargs)
    values.apply_defaults()
    timeout = float(values.arguments.get("timeout_seconds", 30.0))
    if not math.isfinite(timeout):
        raise ValueError("timeout_seconds must be finite")
    return timeout if timeout > 0 else 30.0


def async_tool(fn):
    """Preserve the original public schema and synchronous cross-module calls."""
    @functools.wraps(fn)
    @traced("async_call")
    async def invoke(*args, **kwargs):
        job = None
        try:
            budget = _budget(fn, args, kwargs)
            # Constructing the worker never acquires the model runtime lock.
            runtime = get_worker()
            job = runtime.submit(fn.__name__, lambda: fn(*args, **kwargs), budget)
            future = asyncio.wrap_future(job.future)
            # Consume late failures even when the MCP caller has stopped waiting.
            future.add_done_callback(lambda f: None if f.cancelled() else f.exception())
            try:
                # Nested synchronous workflows may enter a connection phase later.
                # Observe its deadline as well as the original request's deadline.
                while True:
                    with runtime.lock:
                        deadline, expired = job.deadline, job.expired
                    if expired or (deadline is not None and time.monotonic() >= deadline):
                        stage = runtime.expire(job)
                        return failure_result(fn.__name__,
                            f"Total timeout in stage {stage}. The underlying call may still be running; no retry was started.",
                            stage=stage, job=job)
                    wait = 0.05 if deadline is None else max(0, min(0.05, deadline-time.monotonic()))
                    done, _ = await asyncio.wait({future}, timeout=wait)
                    if done:
                        return future.result()
            except asyncio.CancelledError:
                runtime.expire(job)
                raise
        except Exception as exc:
            if isinstance(exc, WorkerBusy):
                job = exc.job
            return failure_result(fn.__name__, exc, stage="dispatch" if job is None else job.stage, job=job)
    # Evaluate annotations in the original module, not this wrapper's globals.
    invoke.__signature__ = inspect.signature(fn, eval_str=True)
    invoke.__annotations__ = {k: v.annotation for k, v in invoke.__signature__.parameters.items()}
    invoke.__annotations__["return"] = invoke.__signature__.return_annotation
    return traced("mcp_tool_entry")(invoke)


def get_worker():
    # Independent of the model lock: a Java call may hold that lock indefinitely.
    with _srv._connection_worker_lock:
        if _srv._connection_worker is None:
            _srv._connection_worker = ToolWorker()
        return _srv._connection_worker


def connection_entry(fn):
    """Serialize legacy direct calls, including background workflow callers."""
    @functools.wraps(fn)
    def invoke(*args, **kwargs):
        parent = current_job()
        if parent is not None:
            runtime = _srv._connection_worker
            with runtime.lock:
                previous = parent.deadline
                deadline = time.monotonic() + _budget(fn, args, kwargs)
                parent.deadline = deadline if previous is None else min(previous, deadline)
            try:
                checkpoint("connection")
                return fn(*args, **kwargs)
            finally:
                with runtime.lock:
                    parent.deadline = previous
        runtime = get_worker()
        job = None
        try:
            budget = _budget(fn, args, kwargs)
            job = runtime.submit(fn.__name__, lambda: fn(*args, **kwargs), budget)
            wait = None if job.deadline is None else max(0, job.deadline - time.monotonic())
            return job.future.result(timeout=wait)
        except concurrent.futures.TimeoutError:
            stage = runtime.expire(job)
            return failure_result(fn.__name__, "Total timeout; underlying call may still be running.", stage=stage, job=job)
        except Exception as exc:
            if isinstance(exc, WorkerBusy):
                job = exc.job
            return failure_result(fn.__name__, exc, stage="dispatch" if job is None else job.stage, job=job)
    return invoke


class AsyncToolRegistry:
    def __init__(self, mcp):
        self.mcp = mcp

    def add_tool(self, fn, **kwargs):
        self.mcp.add_tool(async_tool(fn), **kwargs)
