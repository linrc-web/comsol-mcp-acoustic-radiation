"""Getter-only fake Java tests; never initialize or connect COMSOL."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import comsol_mcp._model as model_module
import comsol_mcp._state as state
import comsol_mcp._tools_solution as tools
from comsol_mcp._server import SAFE_READ_TOOLS


class ReadOnlyJava:
    def __init__(self, **getters):
        self.calls = []
        self.getters = getters

    def __getattr__(self, name):
        # Fail and record even if the tool catches the error. No unknown API
        # (including create/set/remove/run/save/solve) is allowed on any object.
        def call(*args):
            self.calls.append((name, args))
            if name not in self.getters:
                raise AssertionError("Forbidden or unknown API: " + name)
            value = self.getters[name]
            if isinstance(value, Exception):
                raise value
            return value
        return call

    def assert_getters_only(self):
        assert all(name in self.getters for name, _ in self.calls)
        assert all(not name.lower().startswith(
            ("create", "set", "remove", "run", "save", "solve", "clear", "delete"))
            for name, _ in self.calls)
        assert len(self.calls) == len({name for name, _ in self.calls})


@pytest.fixture
def runtime(monkeypatch):
    info = ReadOnlyJava(isValid=True, getLevels=1, getLevelNames=["freq"], getOuterSolnum=[1])
    solution = ReadOnlyJava(isInitialized=True, getPVals=[200000.0, 300000.0],
                            getU=list(range(20)), getUImag=[-1.0] * 20, getSolutioninfo=info)
    java = ReadOnlyJava(sol=solution)
    model = SimpleNamespace(java=java)
    monkeypatch.setattr(model_module, "_require_model", lambda: model)
    monkeypatch.setattr(model_module, "_read_workflow_state", lambda: {})
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error), allow_nan=False))
    yield SimpleNamespace(java=java, solution=solution, info=info)
    for obj in (java, solution, info):
        obj.assert_getters_only()


def call(tag="sol1"):
    return json.loads(tools.inspect_solution(tag))


def test_complete_data_and_bounded_samples(runtime):
    result = call(" sol1 ")
    assert result["success"]
    data = result["data"]
    assert data["is_initialized"] == {"status": "available", "value": True}
    assert data["pvals"] == {"status": "available", "values": [200000.0, 300000.0], "length": 2}
    assert data["u_real"] == {"status": "available", "length": 20,
                              "sample": list(range(8)), "all_finite": True}
    assert data["u_imag"]["sample"] == [-1.0] * 8
    assert data["solution_info"]["level_names"]["values"] == ["freq"]
    assert runtime.java.calls == [("sol", ("sol1",))]
    assert "inspect_solution" in SAFE_READ_TOOLS


@pytest.mark.parametrize("getter,field", [("isInitialized", "is_initialized"),
    ("getPVals", "pvals"), ("getU", "u_real"), ("getUImag", "u_imag"),
    ("getSolutioninfo", "solution_info")])
def test_java_errors_are_independent_and_never_retried(runtime, getter, field):
    runtime.solution.getters[getter] = RuntimeError("COMSOL getter unavailable")
    result = call()
    assert result["success"]
    assert result["data"][field] == {"status": "unavailable", "error": "COMSOL getter unavailable"}
    assert sum(name == getter for name, _ in runtime.solution.calls) == 1
    assert result["data"]["pvals" if field != "pvals" else "u_real"]["status"] == "available"


def test_missing_method_and_partial_metadata(runtime):
    del runtime.solution.getters["getUImag"]
    runtime.info.getters["getLevels"] = AttributeError("not supported")
    result = call()
    # The missing method is intentionally attempted once and reported, not guessed
    # via alternative names. Restore the allowlist for fixture teardown auditing.
    runtime.solution.getters["getUImag"] = AttributeError("not supported")
    assert result["data"]["u_imag"]["status"] == "unavailable"
    info = result["data"]["solution_info"]
    assert info["levels"]["status"] == "unavailable"
    assert info["is_valid"]["value"] is True


def test_empty_and_nonfinite_data(runtime):
    runtime.solution.getters.update(isInitialized=False, getPVals=[], getU=[],
                                    getUImag=[0.0] * 9 + [float("nan"), float("inf")])
    data = call()["data"]
    assert data["is_initialized"]["value"] is False
    assert data["pvals"]["length"] == 0
    assert data["u_real"]["length"] == 0
    assert data["u_imag"]["all_finite"] is False  # scans beyond sample
    assert len(data["u_imag"]["sample"]) == 8


def test_nonfinite_samples_and_parameters_are_valid_json(runtime):
    runtime.solution.getters.update(getPVals=[float("inf")], getU=[float("nan")])
    data = call()["data"]
    assert data["pvals"]["values"] == ["inf"]
    assert data["u_real"]["sample"] == ["nan"]
    assert data["u_real"]["all_finite"] is False


@pytest.mark.parametrize("getter,field", [("getPVals", "pvals"), ("getU", "u_real"),
    ("getUImag", "u_imag"), ("getSolutioninfo", "solution_info")])
def test_null_getter_is_unavailable(runtime, getter, field):
    runtime.solution.getters[getter] = None
    assert call()["data"][field]["status"] == "unavailable"


def test_blank_tag_does_not_access_java(runtime):
    result = call(" ")
    assert not result["success"] and "required" in result["error"]
    assert runtime.java.calls == []


def test_missing_solution_does_not_create_or_retry(runtime):
    runtime.java.getters["sol"] = RuntimeError("Unknown solution: missing")
    result = call("missing")
    assert not result["success"] and "Unknown solution" in result["error"]
    assert runtime.java.calls == [("sol", ("missing",))]
    assert runtime.solution.calls == []


def test_visible_main_guard_is_preserved(runtime, monkeypatch):
    monkeypatch.setattr(model_module, "_read_workflow_state",
                        lambda: {"visible_main_locked": True, "guard_level": "strict"})
    monkeypatch.setattr(model_module, "_visible_main_mismatch", lambda *args: ["tag mismatch"])
    monkeypatch.setattr(model_module, "_visible_main_identity", lambda *args: {})
    monkeypatch.setattr(model_module, "_append_operation", Mock())
    result = call()
    assert not result["success"] and "lock mismatch" in result["error"]
    assert runtime.java.calls == []
