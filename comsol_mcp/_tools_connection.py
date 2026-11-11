#!/usr/bin/env python3
"""MCP tools: server_info, check_server_port, server_start, server_connect, server_disconnect."""

from __future__ import annotations

import json
from typing import Any

import comsol_mcp._server as _srv
from comsol_mcp._state import (
    _run_tool, _run_tool_readonly, _json, _now_iso, _require_mph, _port_is_open,
    _server_missing_guidance, _mark_awaiting_manual_server,
    _write_workflow_state, _read_workflow_state, _write_status,
    _status_payload, _safe_model_label, _resolve_output_path, OUTPUTS_DIR,
    _friendly_connection_error,
)
from comsol_mcp._connection import (
    _disconnect_locked, _ensure_client_shell,
)
from comsol_mcp._model import _set_current_model, _adopt_model_by_name
from comsol_mcp._dispatch import checkpoint, connection_entry


def server_info() -> str:
    """Return COMSOL MCP status and the recommended attach-first workflow.

    For normal visible Desktop usage, manually start COMSOL Multiphysics Server,
    connect Desktop to it, and then call server_connect().
    """

    def _impl() -> dict[str, Any]:
        mph_ok = True
        mph_err = ""
        try:
            _require_mph()
        except Exception as exc:
            mph_ok = False
            mph_err = str(exc)

        loaded_model_count = 0
        if mph_ok and _srv._client is not None and _srv._client_connected:
            try:
                checkpoint("models")
                loaded_model_count = len(list(_srv._client.models()))
            except Exception:
                loaded_model_count = -1

        result: dict[str, Any] = {
            "version": _srv.VERSION,
            "comsol_root": str(_srv.COMSOL_ROOT),
            "mcp_home": str(_srv.COMSOL_SERVER_MCP_HOME),
            "workflow_file": str(_srv.WORKFLOW_FILE),
            "status_file": str(_srv.STATUS_FILE),
            "recommended_entrypoint": 'server_connect("localhost", <actual_port>)',
            "server_start_role": "advanced/manual-lifecycle",
            "server_state": _status_payload(),
            "loaded_model_count": loaded_model_count,
        }
        if mph_ok:
            result["mph_version"] = getattr(_srv._get_mph(), "__version__", "")
            result["mphserver_exe"] = str(_srv.COMSOL_MPHSERVER)
            result["mphclient_exe"] = str(_srv.COMSOL_MPHCLIENT)
        else:
            result["mph_available"] = False
            result["mph_error"] = mph_err
            result["message"] = "MPh is not available. Install it or check Python environment."
        if loaded_model_count < 0:
            result["connection_warning"] = "Could not enumerate models — server connection may be stale."
        return result

    return _run_tool_readonly("server_info", _impl)


def check_server_port(host: str = _srv.DEFAULT_HOST, port: int = _srv.DEFAULT_PORT) -> str:
    """Check whether a manually started COMSOL Server is listening on host:port."""

    def _impl() -> dict[str, Any]:
        requested_host = host or _srv.DEFAULT_HOST
        requested_port = int(port)
        available = _port_is_open(requested_host, requested_port)
        if not available:
            workflow = _mark_awaiting_manual_server(requested_host, requested_port, "check_server_port")
            return {**_server_missing_guidance(requested_host, requested_port), "workflow": workflow}
        workflow = _write_workflow_state(
            {
                "last_action": "check_server_port",
                "last_checked_host": requested_host,
                "last_checked_port": requested_port,
                "last_checked_at": _now_iso(),
                "workflow_stage": "manual_server_available",
            }
        )
        return {
            "server_available": True,
            "host": requested_host,
            "port": requested_port,
            "message": (
                f"COMSOL Multiphysics Server is listening at {requested_host}:{requested_port}. "
                "Next MCP step: start_visible_main_workflow(host, port, path)."
            ),
            "next_step_for_mcp": "Call start_visible_main_workflow(...) to connect and load the main MPH.",
            "workflow": workflow,
        }

    return _run_tool_readonly("check_server_port", _impl)


@connection_entry
def server_start(
    port: int = 0,
    cores: int = 0,
    multi: str = "on",
    timeout_seconds: int = _srv.DEFAULT_TIMEOUT,
    version: str = "",
) -> str:
    """Advanced fallback only: start a local COMSOL server and connect MCP to it.

    Do NOT use this as the default visible workflow.
    Preferred workflow:
    1. Manually start COMSOL Multiphysics Server.
    2. Connect COMSOL Desktop to that server.
    3. Call server_connect(host, port) from MCP.

    Use server_start() only when MCP should own the COMSOL server lifecycle and
    random or non-default listening ports are acceptable.
    """

    def _impl() -> dict[str, Any]:
        _require_mph()
        _disconnect_locked(shutdown_server=True)
        effective_version = version.strip() or _srv.DEFAULT_VERSION
        effective_port = int(port)
        effective_cores = int(cores) if int(cores) > 0 else None
        _srv._server = _srv._get_mph().Server(
            cores=effective_cores,
            version=effective_version,
            port=effective_port,
            multi=multi or "on",
            timeout=max(5, int(timeout_seconds)),
        )
        _srv._server_started_by_mcp = True
        client = _ensure_client_shell(effective_version)
        checkpoint("connect")
        _srv._client_connection_uncertain = True
        client.connect(_srv._server.port, _srv.DEFAULT_HOST)
        checkpoint("connected")
        _srv._client_connected = True
        _srv._client_connection_uncertain = False
        _srv._connected_host = _srv.DEFAULT_HOST
        _srv._connected_port = _srv._server.port
        _write_status()
        return {
            "started_by_mcp": True,
            "attached_to_existing_server": False,
            "host": _srv.DEFAULT_HOST,
            "port": _srv._server.port,
            "endpoint": f"{_srv.DEFAULT_HOST}:{_srv._server.port}",
            "version": effective_version or "",
            "recommended_entrypoint": 'server_connect("localhost", <actual_port>)',
            "recommended_usage": (
                "Prefer manual COMSOL Multiphysics Server startup for visible Desktop "
                "workflows. Use server_start() only when MCP should own the server lifecycle "
                "and accept that COMSOL may choose a different listening port."
            ),
            "desktop_should_connect": {"host": _srv.DEFAULT_HOST, "port": _srv._server.port},
        }

    return _run_tool("server_start", _impl)


@connection_entry
def server_connect(host: str = _srv.DEFAULT_HOST, port: int = _srv.DEFAULT_PORT, model_name: str = "", timeout_seconds: float = 30.0) -> str:
    """Default entrypoint: attach MCP to an already running COMSOL Multiphysics Server.

    This is the recommended tool for the visible Desktop workflow:
    manually start COMSOL Multiphysics Server, call server_connect(host, port),
    load_visible_main_model(path), then connect Desktop and import that model.

    timeout_seconds covers the entire operation, including queue/lock wait,
    import, JVM initialization, connection and model enumeration. A timeout
    stops waiting, not the underlying Java call; overlapping calls are rejected.
    """

    def _impl() -> dict[str, Any]:
        checkpoint("import")
        _require_mph()
        checkpoint("port")
        requested_host = host or _srv.DEFAULT_HOST
        requested_port = int(port)

        # Pre-check: TCP port probe before attempting the heavy mph connection
        if not _port_is_open(requested_host, requested_port, timeout_seconds=2.0):
            _mark_awaiting_manual_server(requested_host, requested_port, "server_connect")
            return {
                "connected": False,
                "host": requested_host,
                "port": requested_port,
                "error": (
                    f"Port {requested_port} on {requested_host} is not open. "
                    "COMSOL Multiphysics Server is likely not running or listening on a different port."
                ),
                "suggestion": "Start COMSOL Server manually and verify the port number before retrying.",
            }
        preserve_local_server = bool(
            _srv._server is not None
            and _srv._server_started_by_mcp
            and getattr(_srv._server, "port", None) == requested_port
            and requested_host == _srv.DEFAULT_HOST
        )
        if _srv._client is not None:
            _disconnect_locked(shutdown_server=False)
        if _srv._server is not None and _srv._server_started_by_mcp and not preserve_local_server:
            _disconnect_locked(shutdown_server=True)
        elif _srv._server is None:
            _srv._server_started_by_mcp = False
        client = _ensure_client_shell()
        try:
            checkpoint("connect")
            _srv._client_connection_uncertain = True
            client.connect(requested_port, requested_host)
            checkpoint("connected")
        except Exception as exc:
            raise _friendly_connection_error(exc, requested_host, requested_port) from exc
        _srv._client_connected = True
        _srv._client_connection_uncertain = False
        _srv._connected_host = requested_host
        _srv._connected_port = requested_port
        checkpoint("models")
        adopted = _adopt_model_by_name(model_name.strip())
        checkpoint("adopt")
        if adopted is not None:
            _set_current_model(adopted, origin="connected")
        workflow = _read_workflow_state()
        if not bool(workflow.get("visible_main_locked")):
            _write_workflow_state({"workflow_stage": "mcp_connected"})
        _write_status()
        current_label = _safe_model_label(_srv._current_model)
        return {
            "connected": True,
            "attached_to_existing_server": True,
            "host": requested_host,
            "port": requested_port,
            "endpoint": f"{requested_host}:{requested_port}",
            "current_model_label": current_label,
            "has_current_model": bool(current_label),
            "next_step": (
                'Use model_create() or model_load() to select a working model.'
                if not current_label
                else ""
            ),
            "preserved_local_server": preserve_local_server,
        }

    return _run_tool("server_connect", _impl)


@connection_entry
def server_disconnect(shutdown_server: bool = False) -> str:
    """Disconnect the MCP client. Optionally shut down the local server if MCP started it."""

    def _impl() -> dict[str, Any]:
        connected = bool(_srv._client_connected)
        started_by_mcp = _srv._server_started_by_mcp
        _disconnect_locked(shutdown_server=bool(shutdown_server))
        _write_status()
        return {
            "disconnected": connected,
            "shutdown_server": bool(shutdown_server),
            "server_was_started_by_mcp": started_by_mcp,
            "disconnect_mode": (
                "stopped-mcp-started-server"
                if bool(shutdown_server) and started_by_mcp
                else "detached-from-server"
            ),
        }

    return _run_tool("server_disconnect", _impl)


def register(mcp_instance) -> None:
    mcp_instance.add_tool(server_info)
    mcp_instance.add_tool(check_server_port)
    mcp_instance.add_tool(server_start)
    mcp_instance.add_tool(server_connect)
    mcp_instance.add_tool(server_disconnect)
