"""Synchronous study execution tests using mocks; no COMSOL or JVM startup."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import comsol_mcp._state as state
import comsol_mcp._tools_workflow as tools


@pytest.fixture
def runtime(monkeypatch):
    model = Mock()
    model.java.study.return_value.get.return_value.label.return_value = "Study 1"
    guard = Mock(return_value=model)
    monkeypatch.setattr(tools, "_require_visible_main", guard)
    monkeypatch.setattr(tools, "_safe_model_label", lambda model: "Excited.mph")
    # Keep the actual _run_tool and JSON result envelope; replace external IO.
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_status_payload", lambda: {"status": "ready"})
    monkeypatch.setattr(state, "_append_operation", lambda *a: None)
    monkeypatch.setattr(state, "_write_status", lambda *a: None)
    monkeypatch.setattr(state, "current_job", lambda: None)
    monkeypatch.setattr(state.logging, "exception", Mock())
    return SimpleNamespace(model=model, guard=guard)


@pytest.mark.parametrize("study_tag", [" study1 ", "", "   "])
@pytest.mark.parametrize("failure", [False, True])
def test_solve_is_called_once_and_result_format_preserved(runtime, study_tag, failure):
    if failure:
        runtime.model.solve.side_effect = RuntimeError("First solve failed")
    result = json.loads(tools.run_study(study_tag=study_tag))
    if study_tag.strip():
        runtime.model.java.study.return_value.get.assert_called_once_with("study1")
        runtime.model.solve.assert_called_once_with("Study 1")
    else:
        runtime.model.java.study.assert_not_called()
        runtime.model.solve.assert_called_once_with()
    runtime.guard.assert_called_once_with("run_study")
    assert result["tool"] == "run_study"
    assert result["success"] is (not failure)
    assert result["error"] == ("First solve failed" if failure else "")
    assert result["data"] == ({} if failure else {
        "study_tag": study_tag.strip(), "label": "Excited.mph"})
    assert set(result) == {"success", "tool", "timestamp", "datetime", "server",
                           "data", "error", "log_path", "operations_path"}


def test_default_signature_solves_without_argument(runtime):
    assert json.loads(tools.run_study())["success"]
    runtime.model.solve.assert_called_once_with()


def test_label_lookup_failure_does_not_attempt_solve(runtime):
    runtime.model.java.study.return_value.get.return_value.label.side_effect = RuntimeError("Study lookup failed")
    result = json.loads(tools.run_study("study1"))
    assert not result["success"]
    assert result["error"] == "Study lookup failed"
    assert result["data"] == {}
    runtime.model.solve.assert_not_called()


def test_guard_failure_does_not_attempt_solve(runtime):
    runtime.guard.side_effect = RuntimeError("Visible main lock mismatch")
    result = json.loads(tools.run_study("study1"))
    assert not result["success"]
    assert result["error"] == "Visible main lock mismatch"
    runtime.model.solve.assert_not_called()
