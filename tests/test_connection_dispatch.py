"""Connection lifecycle tests: fake MPH only, no Java or COMSOL sockets."""
import asyncio
import inspect
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import comsol_mcp._server as srv
import comsol_mcp._state as state
import comsol_mcp._connection as connection
import comsol_mcp._tools_connection as tools
from comsol_mcp._dispatch import async_tool, get_worker

_real_get_mph = srv._get_mph


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    events = []
    hooks = {}
    release = threading.Event()

    def record(stage):
        events.append((stage, threading.get_ident()))
        if stage in hooks:
            hooks[stage]()

    class Client:
        def __init__(self, **kwargs):
            record("initialize")
            self.version = "6.4"

        def connect(self, port, host):
            record("connect")

        def models(self):
            record("models")
            return []

        def disconnect(self):
            record("disconnect")

    fake = SimpleNamespace(Client=Client)

    def get_mph():
        record("import")
        return fake

    for attr, value in {"_client": None, "_client_connected": False,
                        "_client_initialization_attempted": False,
                        "_client_connection_uncertain": False,
                        "_connection_worker": None, "_server": None,
                        "_server_started_by_mcp": False, "_current_model": None,
                        "_connected_host": "", "_connected_port": None,
                        "_runtime_lock": threading.RLock()}.items():
        monkeypatch.setattr(srv, attr, value)
    monkeypatch.setattr(state, "_runtime_lock", srv._runtime_lock)
    monkeypatch.setattr(state, "_get_mph", get_mph)
    monkeypatch.setattr(connection, "_get_mph", get_mph)
    monkeypatch.setattr(srv, "_get_mph", get_mph)
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(tools, "_port_is_open", lambda *a, **kw: True)
    monkeypatch.setattr(tools, "_read_workflow_state", lambda: {})
    monkeypatch.setattr(tools, "_write_workflow_state", lambda update: update)
    monkeypatch.setattr(tools, "_write_status", lambda *a: None)
    monkeypatch.setattr(state, "_status_payload", lambda: {"status": "test"})
    monkeypatch.setattr(state, "_append_operation", lambda *a: None)
    monkeypatch.setattr(state, "_write_status", lambda *a: None)
    before = set(sys.modules)
    yield SimpleNamespace(events=events, hooks=hooks, release=release)
    release.set()
    runtime = srv._connection_worker
    if runtime is not None:
        runtime.close()
        assert not runtime.thread.is_alive()
    assert not ({"mph", "jpype", "_jpype"} & (set(sys.modules) - before))


def call_connect(timeout=1):
    return async_tool(tools.server_connect)(host="localhost", port=2036, timeout_seconds=timeout)


def test_success_reuses_client_and_one_worker(fake_runtime):
    async def scenario():
        for _ in range(2):
            result = json.loads(await call_connect())
            assert result["success"] and result["data"]["connected"]
            result = json.loads(await async_tool(tools.server_disconnect)())
            assert result["success"] and result["data"]["disconnected"]
    asyncio.run(scenario())
    names = [name for name, _ in fake_runtime.events]
    assert names.count("initialize") == 1
    assert names.count("connect") == names.count("models") == names.count("disconnect") == 2
    assert len({tid for _, tid in fake_runtime.events}) == 1
    assert fake_runtime.events[0][1] != threading.get_ident()
    assert srv._client is not None


@pytest.mark.parametrize("stage,reported", [("import", "import"), ("initialize", "initialize"),
                                          ("connect", "connect"), ("models", "models")])
def test_total_timeout_keeps_event_loop_responsive_and_blocks_overlap(fake_runtime, stage, reported):
    entered = threading.Event()
    def block():
        entered.set()
        fake_runtime.release.wait(2)
    fake_runtime.hooks[stage] = block

    async def scenario():
        task = asyncio.create_task(call_connect(0.08))
        ticks = 0
        while not task.done():
            ticks += 1
            await asyncio.sleep(0.005)
        result = json.loads(await task)
        assert entered.is_set()
        assert ticks > 2
        assert not result["success"]
        assert result["data"]["stage"] == reported
        assert result["data"]["operation_pending"]
        events = list(fake_runtime.events)
        retry = json.loads(await call_connect())
        assert not retry["success"] and "overlapping" in retry["error"]
        assert retry["data"]["operation_pending"]
        assert retry["data"]["stage"] == reported
        assert events == fake_runtime.events
        fake_runtime.release.set()
        while get_worker().pending:
            await asyncio.sleep(0.005)
        assert not srv._client_connected
        # A late return must not continue to the next stage.
        names = [name for name, _ in fake_runtime.events]
        if stage == "initialize":
            assert "connect" not in names
        if stage == "connect":
            assert "models" not in names
    asyncio.run(scenario())


def test_lock_timeout_does_not_import_mph(fake_runtime):
    srv._runtime_lock.acquire()
    try:
        result = json.loads(asyncio.run(call_connect(0.05)))
        assert not result["success"]
        assert result["data"]["stage"] == "lock"
        assert not fake_runtime.events
    finally:
        srv._runtime_lock.release()


@pytest.mark.parametrize("stage", ["initialize", "connect", "models", "disconnect"])
def test_errors_return_original_message(fake_runtime, stage):
    def fail():
        raise RuntimeError("simulated failure: " + stage)
    fake_runtime.hooks[stage] = fail
    async def scenario():
        result = json.loads(await call_connect())
        if stage == "disconnect":
            assert result["success"]
            result = json.loads(await async_tool(tools.server_disconnect)())
        assert not result["success"]
        assert "simulated failure: " + stage in result["error"]
        if stage == "initialize":
            result = json.loads(await call_connect())
            assert "restart" in result["error"]
            assert [s for s, _ in fake_runtime.events].count("initialize") == 1
    asyncio.run(scenario())


def test_schema_and_mcp_adapter(fake_runtime):
    from comsol_mcp._server import mcp
    import comsol_mcp.mcp_server
    registered = mcp._tool_manager._tools["server_connect"]
    assert registered.is_async
    assert inspect.signature(registered.fn) == inspect.signature(tools.server_connect, eval_str=True)
    async def scenario():
        result = json.loads(await registered.run({"host": "localhost", "port": 2036}))
        assert result["success"]
    asyncio.run(scenario())


def test_direct_synchronous_call_uses_same_worker(fake_runtime):
    result = json.loads(tools.server_connect(timeout_seconds=1))
    assert result["success"]
    assert len({tid for _, tid in fake_runtime.events}) == 1
    assert fake_runtime.events[0][1] != threading.get_ident()


def test_timeout_result_never_reads_live_status(fake_runtime, monkeypatch):
    def forbidden():
        raise AssertionError("must not read live server/model state")
    monkeypatch.setattr(state, "_status_payload", forbidden)
    fake_runtime.hooks["connect"] = lambda: fake_runtime.release.wait(2)
    result = json.loads(asyncio.run(call_connect(0.05)))
    assert not result["success"]
    assert result["server"]["state_confirmed"] is False
    assert result["data"]["stage"] == "connect"


def test_budget_is_shared_across_phases(fake_runtime):
    fake_runtime.hooks["initialize"] = lambda: time.sleep(0.04)
    fake_runtime.hooks["connect"] = lambda: fake_runtime.release.wait(2)
    started = time.monotonic()
    result = json.loads(asyncio.run(call_connect(0.07)))
    assert not result["success"]
    assert result["data"]["stage"] == "connect"
    assert time.monotonic() - started < 0.3


def test_cancelled_request_keeps_worker_owned_until_return(fake_runtime):
    entered = threading.Event()
    def block():
        entered.set()
        fake_runtime.release.wait(2)
    fake_runtime.hooks["connect"] = block
    async def scenario():
        task = asyncio.create_task(call_connect())
        while not entered.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = json.loads(await call_connect())
        assert not result["success"] and "overlapping" in result["error"]
        with pytest.raises(RuntimeError, match="pending"):
            connection._require_client()
    asyncio.run(scenario())


def test_recover_after_late_call_returns_reuses_same_client(fake_runtime):
    fake_runtime.hooks["connect"] = lambda: fake_runtime.release.wait(2)
    async def scenario():
        result = json.loads(await call_connect(0.05))
        assert not result["success"]
        fake_runtime.release.set()
        while get_worker().pending:
            await asyncio.sleep(0.005)
        fake_runtime.hooks.clear()
        result = json.loads(await call_connect())
        assert result["success"]
    asyncio.run(scenario())
    names = [name for name, _ in fake_runtime.events]
    assert names.count("initialize") == 1
    assert names.index("disconnect") < len(names) - 1


def test_queue_wait_is_part_of_total_budget(fake_runtime):
    runtime = get_worker()
    blocker = runtime.submit("unrelated", lambda: fake_runtime.release.wait(2))
    result = json.loads(asyncio.run(call_connect(0.05)))
    assert not result["success"]
    assert result["data"]["stage"] == "queued"
    fake_runtime.release.set()
    blocker.future.result(timeout=1)
    runtime.close()
    assert not fake_runtime.events


def test_nested_workflow_connection_has_deadline(fake_runtime, monkeypatch):
    from comsol_mcp import _dispatch
    original = _dispatch._budget
    monkeypatch.setattr(_dispatch, "_budget", lambda fn, a, kw:
                        0.05 if fn.__name__ == "server_connect" else original(fn, a, kw))
    fake_runtime.hooks["connect"] = lambda: fake_runtime.release.wait(2)
    def workflow():
        return tools.server_connect()
    result = json.loads(asyncio.run(async_tool(workflow)()))
    assert not result["success"]
    assert result["data"]["stage"] == "connect"


def test_disconnect_timeout_uses_same_worker(fake_runtime, monkeypatch):
    from comsol_mcp import _dispatch
    original = _dispatch._budget
    monkeypatch.setattr(_dispatch, "_budget", lambda fn, a, kw:
                        0.05 if fn.__name__ == "server_disconnect" else original(fn, a, kw))
    fake_runtime.hooks["disconnect"] = lambda: fake_runtime.release.wait(2)
    async def scenario():
        assert json.loads(await call_connect())["success"]
        result = json.loads(await async_tool(tools.server_disconnect)())
        assert not result["success"] and result["data"]["stage"] == "disconnect"
    asyncio.run(scenario())
    assert len({tid for _, tid in fake_runtime.events}) == 1


def test_nested_timed_call_never_creates_another_java_thread(fake_runtime):
    def workflow():
        return connection._timed_call(lambda: threading.get_ident(), timeout=1)
    tid = asyncio.run(async_tool(workflow)())
    assert tid == get_worker().thread.ident


def test_nested_timed_call_deadline_keeps_operation_pending(fake_runtime):
    def blocked_model_read():
        return fake_runtime.release.wait(2)
    def workflow():
        return connection._timed_call(blocked_model_read, timeout=0.05)
    async def scenario():
        result = json.loads(await async_tool(workflow)())
        assert not result["success"]
        assert result["data"]["stage"] == "blocked_model_read"
        result = json.loads(await call_connect())
        assert not result["success"] and "overlapping" in result["error"]
    asyncio.run(scenario())

def test_connection_diagnostics(fake_runtime, capsys, monkeypatch):
    import builtins
    import os
    import re
    # Exercise the real lazy import with an import hook, never loading MPH/Java.
    original_import = builtins.__import__
    fake_mph = connection._get_mph()
    def fake_import(name, *args, **kwargs):
        if name == "mph":
            return fake_mph
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(srv, "_mph", None)
    _get_mph = _real_get_mph
    monkeypatch.setattr(state, "_get_mph", _get_mph)
    monkeypatch.setattr(connection, "_get_mph", _get_mph)
    result = json.loads(asyncio.run(call_connect()))
    assert result["success"]
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    for stage in ("mcp_tool_entry", "async_call", "worker_start", "worker_call",
                  "model_lock_acquire", "mph_import", "jvm_initialization"):
        for event in ("start", "end"):
            matching = [line for line in lines if f"stage={stage} event={event}" in line]
            assert len(matching) == 1
            assert f"pid={os.getpid()}" in matching[0]
            assert re.search(r"thread_id=\d+", matching[0])
            assert re.match(r"\d{4}-\d\d-\d\dT", matching[0])


def test_diagnostics_exception_and_broken_stderr(capsys, monkeypatch):
    from comsol_mcp._diagnostics import phase
    with pytest.raises(ValueError, match="secret-password"):
        with phase("test_failure"):
            raise ValueError("secret-password")
    captured = capsys.readouterr()
    assert "event=start" in captured.err and "event=end" in captured.err
    assert "secret-password" not in captured.err
    class BrokenStderr:
        def write(self, text):
            raise OSError("unavailable")
    monkeypatch.setattr(sys, "stderr", BrokenStderr())
    with phase("test_failure"):
        pass


@pytest.mark.parametrize("setting", [None, "0", "true"])
def test_main_preinit_disabled(fake_runtime, monkeypatch, setting):
    import comsol_mcp.mcp_server as entry
    if setting is None:
        monkeypatch.delenv("COMSOL_MCP_PREINIT_CLIENT", raising=False)
    else:
        monkeypatch.setenv("COMSOL_MCP_PREINIT_CLIENT", setting)
    calls = []
    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_write_status", lambda value: calls.append("ready"))
    monkeypatch.setattr(entry.mcp, "run", lambda: calls.append("run"))
    entry.main()
    assert calls == ["ready", "run"]
    assert fake_runtime.events == []
    assert srv._client is None
    assert srv._connection_worker is None


def test_main_preinit_reuses_shared_client_on_worker(fake_runtime, monkeypatch):
    import comsol_mcp.mcp_server as entry
    monkeypatch.setenv("COMSOL_MCP_PREINIT_CLIENT", "1")
    monkeypatch.setattr(srv, "DEFAULT_VERSION", "6.4")
    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    calls = []
    def ready(value):
        assert srv._client is not None
        assert srv._client_initialization_attempted
        assert not srv._client_connected
        assert srv._connection_worker is None
        calls.append("ready")
    monkeypatch.setattr(entry, "_write_status", ready)
    monkeypatch.setattr(entry.mcp, "run", lambda: calls.append("run"))
    entry.main()
    client = srv._client
    assert client.version == "6.4"
    assert calls == ["ready", "run"]
    assert fake_runtime.events == [("import", threading.get_ident()),
                                   ("initialize", threading.get_ident())]
    # Exercise reuse on the real dispatch worker with a fake MPH client only.
    assert asyncio.run(async_tool(connection._ensure_client_shell)()) is client
    assert srv._client is client
    assert [name for name, _ in fake_runtime.events].count("initialize") == 1
    assert srv._connection_worker.thread.ident != threading.get_ident()


def test_main_preinit_failure_stops_startup(fake_runtime, monkeypatch):
    import comsol_mcp.mcp_server as entry
    monkeypatch.setenv("COMSOL_MCP_PREINIT_CLIENT", "1")
    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    def forbidden(*args):
        pytest.fail("must not report ready or start MCP after initialization failure")
    monkeypatch.setattr(entry, "_write_status", forbidden)
    monkeypatch.setattr(entry.mcp, "run", forbidden)
    def fail():
        raise RuntimeError("simulated startup failure")
    fake_runtime.hooks["initialize"] = fail
    with pytest.raises(RuntimeError, match="simulated startup failure"):
        entry.main()
    assert srv._client is None
    assert srv._client_initialization_attempted
    assert srv._connection_worker is None
