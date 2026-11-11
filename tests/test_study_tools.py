"""Study configuration tests: fake Java only; no COMSOL connection."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import comsol_mcp._model as model_module
import comsol_mcp._state as state
import comsol_mcp._tools_solver as tools
from comsol_mcp._server import SAFE_VISIBLE_MAIN_WRITE_TOOLS


class Step:
    def __init__(self, kind):
        self.kind = kind
        self.set = Mock()

    def type(self):
        return self.kind


class Study:
    def __init__(self):
        self.steps = {}
        self.create = Mock(side_effect=self._create)

    def _create(self, tag, kind):
        self.steps[tag] = Step(kind)

    def feature(self, tag=None):
        return SimpleNamespace(tags=lambda: list(self.steps)) if tag is None else self.steps[tag]


@pytest.fixture
def runtime(monkeypatch):
    studies = {}
    create = Mock(side_effect=lambda tag: studies.__setitem__(tag, Study()))
    def study(tag=None):
        return SimpleNamespace(tags=lambda: list(studies), create=create) if tag is None else studies[tag]
    model = SimpleNamespace(java=SimpleNamespace(study=study))
    # Exercise the real guard and _run_tool; replace only external state/log IO.
    monkeypatch.setattr(model_module, "_require_model", lambda: model)
    monkeypatch.setattr(model_module, "_read_workflow_state", lambda: {})
    monkeypatch.setattr(model_module, "_append_operation", lambda *a: None)
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error)))
    return SimpleNamespace(model=model, studies=studies, create=create)


def call(name, *args):
    return json.loads(getattr(tools, name)(*args))


def prepared(runtime):
    assert call("create_study", "study1")["success"]
    assert call("create_frequency_domain_step", "study1", "freq")["success"]
    return runtime.studies["study1"].steps["freq"]


def test_normal_and_duplicate_creation(runtime):
    first = call("create_study", " study1 ")
    assert first["data"] == {"study_tag": "study1", "created": True}
    assert call("create_study", "study1")["data"]["created"] is False
    runtime.create.assert_called_once_with("study1")
    assert call("create_frequency_domain_step", "study1", "freq")["data"]["created"] is True
    assert call("create_frequency_domain_step", "study1", "freq")["data"]["created"] is False
    runtime.studies["study1"].create.assert_called_once_with("freq", "Frequency")
    result = call("set_study_frequency", "study1", "freq", " 200[kHz] ")
    assert result["success"] and result["data"]["frequency"] == "200[kHz]"
    runtime.studies["study1"].steps["freq"].set.assert_called_once_with("plist", "200[kHz]")


@pytest.mark.parametrize("name,args", [
    ("create_study", (" ",)),
    ("create_frequency_domain_step", ("", "freq")),
    ("create_frequency_domain_step", ("study1", "")),
    ("set_study_frequency", ("study1", "freq", "")),
    ("set_study_frequency", ("", "freq", "200[kHz]")),
    ("set_study_frequency", ("study1", "", "200[kHz]")),
])
def test_empty_input(runtime, name, args):
    result = call(name, *args)
    assert not result["success"] and "required" in result["error"]
    runtime.create.assert_not_called()


def test_missing_nodes_and_wrong_type(runtime):
    assert "does not exist" in call("create_frequency_domain_step", "missing", "freq")["error"]
    assert "does not exist" in call("set_study_frequency", "missing", "freq", "200[kHz]")["error"]
    call("create_study", "study1")
    assert "does not exist" in call("set_study_frequency", "study1", "freq", "200[kHz]")["error"]
    wrong = Step("Stationary")
    runtime.studies["study1"].steps["freq"] = wrong
    for name, args in [("create_frequency_domain_step", ("study1", "freq")),
                       ("set_study_frequency", ("study1", "freq", "200[kHz]"))]:
        result = call(name, *args)
        assert not result["success"] and "not a Frequency" in result["error"]
    wrong.set.assert_not_called()
    runtime.studies["study1"].create.assert_not_called()


@pytest.mark.parametrize("name", ["create_study", "create_frequency_domain_step", "set_study_frequency"])
def test_java_errors_not_retried(runtime, name):
    if name == "create_study":
        target, args = runtime.create, ("study1",)
    elif name == "create_frequency_domain_step":
        call("create_study", "study1")
        target, args = runtime.studies["study1"].create, ("study1", "freq")
    else:
        target, args = prepared(runtime).set, ("study1", "freq", "200[kHz]")
    target.side_effect = RuntimeError("Java failure detail")
    result = call(name, *args)
    assert not result["success"] and result["error"] == "Java failure detail"
    assert target.call_count == 1


@pytest.mark.parametrize("name,args", [
    ("create_study", ("study1",)),
    ("create_frequency_domain_step", ("study1", "freq")),
    ("set_study_frequency", ("study1", "freq", "200[kHz]")),
])
def test_visible_main_mismatch_blocks_writes(runtime, monkeypatch, name, args):
    step = prepared(runtime)
    runtime.create.reset_mock()
    runtime.studies["study1"].create.reset_mock()
    monkeypatch.setattr(model_module, "_read_workflow_state", lambda: {"visible_main_locked": True, "guard_level": "strict"})
    monkeypatch.setattr(model_module, "_visible_main_mismatch", lambda *a: ["tag mismatch"])
    monkeypatch.setattr(model_module, "_visible_main_identity", lambda *a: {})
    result = call(name, *args)
    assert not result["success"] and "lock mismatch" in result["error"]
    assert name in SAFE_VISIBLE_MAIN_WRITE_TOOLS
    runtime.create.assert_not_called()
    runtime.studies["study1"].create.assert_not_called()
    step.set.assert_not_called()
