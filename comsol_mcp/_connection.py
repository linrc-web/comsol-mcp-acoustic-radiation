#!/usr/bin/env python3
"""Connection lifecycle: disconnect, client shell management, require_client."""

from __future__ import annotations

import concurrent.futures
from typing import Any

from comsol_mcp._diagnostics import phase

from comsol_mcp._server import _get_mph
import comsol_mcp._server as _srv
from comsol_mcp._dispatch import checkpoint, current_job, get_worker


def _timed_call(func, *args, timeout: float = 30.0, error_msg=""):
    """Run on the same MPH worker; a deadline never terminates a Java call.

    Nested calls lend their deadline to the asynchronous MCP waiter. Legacy
    background callers wait on that worker instead of creating another thread.
    """
    runtime = get_worker()
    parent = current_job()
    if parent is not None:
        import time
        with runtime.lock:
            previous = parent.deadline
            deadline = time.monotonic() + timeout
            parent.deadline = deadline if previous is None else min(previous, deadline)
        try:
            checkpoint(getattr(func, "__name__", "java_call"))
            result = func(*args)
            checkpoint("java_call_returned")
            return result
        finally:
            with runtime.lock:
                parent.deadline = previous
    job = runtime.submit(getattr(func, "__name__", "java_call"), lambda: func(*args), timeout)
    try:
        return job.future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        runtime.expire(job)
        raise RuntimeError(error_msg or f"Operation timed out after {timeout}s; underlying call may still be running.")


def _disconnect_locked(*, shutdown_server: bool = False) -> None:
    if _srv._client is not None and (_srv._client_connected or _srv._client_connection_uncertain):
        try:
            checkpoint("disconnect")
            _srv._client.disconnect()
            _srv._client_connection_uncertain = False
        except Exception as exc:
            if "not connected" not in str(exc).lower():
                _srv._last_error = f"Disconnect failed: {exc}"
                _srv._client_connection_uncertain = True
                raise
            _srv._client_connection_uncertain = False
        finally:
            _srv._client_connected = False
            _srv._connected_host = ""
            _srv._connected_port = None

    _srv._current_model = None
    _srv._current_model_origin = ""
    _srv._current_model_path = ""

    if shutdown_server and _srv._server is not None and _srv._server_started_by_mcp:
        try:
            _srv._server.stop()
        except Exception as exc:
            _srv._last_error = f"Server stop failed: {exc}"
        finally:
            _srv._server = None
            _srv._server_started_by_mcp = False
    elif shutdown_server:
        _srv._server = None
        _srv._server_started_by_mcp = False


def _ensure_client_shell(version: str | None = None) -> Any:
    requested_version = version or _srv.DEFAULT_VERSION
    if _srv._client is not None and requested_version and _srv._client.version != requested_version:
        raise RuntimeError("The JVM is already initialized for a different COMSOL version; restart MCP to change versions.")
    if _srv._client is None:
        checkpoint("import")
        mph = _get_mph()
        if mph is None:
            raise RuntimeError("MPh is not available; cannot create client shell.")
        if _srv._client_initialization_attempted:
            raise RuntimeError("Client initialization previously failed; restart the MCP process before retrying.")
        checkpoint("initialize")
        _srv._client_initialization_attempted = True
        with phase("jvm_initialization"):
            _srv._client = mph.Client(version=requested_version, host=None)
        checkpoint("initialized")
    return _srv._client


def _require_client() -> Any:
    runtime = _srv._connection_worker
    if runtime is not None:
        with runtime.lock:
            if any(job.expired for job in runtime.pending):
                raise RuntimeError("A timed-out operation is still pending; client state is unconfirmed.")
    if _srv._client is None or not _srv._client_connected:
        raise RuntimeError("Not connected to a COMSOL Multiphysics Server.")
    return _srv._client
