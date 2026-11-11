"""Topology tests use mocks only; never initialize a COMSOL client."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import comsol_mcp._state as state
import comsol_mcp._tools_geometry as tools
from comsol_mcp._server import SAFE_READ_TOOLS


VALUES = {
    "getSDim": 3, "getNDomains": 1, "getNBoundaries": 12,
    "getNEntities": [16, 24, 12, 1],
    "getBoundingBox": [-0.75, 0.75, -0.1, 0.1, -0.1, 0.1],
    "getNFiniteVoids": 1, "voidsAreLabeled": True,
    "getUpDownExt": [[1, 1, 1], [0, -1, -2]],
}


def info_mock(object_scope=False):
    # spec_set rejects every non-allowlisted API, including mutation and measure.
    info = Mock(spec_set=list(VALUES) + (["exists"] if object_scope else ["obj"]))
    for method, value in VALUES.items():
        getattr(info, method).return_value = value
    if object_scope:
        info.exists.return_value = True
    return info


@pytest.fixture
def runtime(monkeypatch):
    geom = info_mock()
    obj = info_mock(True)
    geom.obj.return_value = obj
    model = SimpleNamespace(java=Mock(spec_set=["component"]))
    model.java.component.return_value.geom.return_value = geom
    require = Mock(return_value=model)
    monkeypatch.setattr(tools, "_require_visible_main", require)
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error)))
    return SimpleNamespace(geom=geom, obj=obj, require=require, model=model)


def invoke(object_name=None, **kwargs):
    return json.loads(tools.get_geometry_topology_info(
        kwargs.get("component", "comp1"), kwargs.get("geometry", "geom1"), object_name))


@pytest.mark.parametrize("object_name", [None, "fluid1"])
def test_raw_topology_and_exact_getter_allowlist(runtime, object_name):
    info = runtime.geom if object_name is None else runtime.obj
    result = invoke(object_name)
    assert result["success"]
    assert result["data"] == {
        "component": "comp1", "geometry": "geom1",
        "scope": "final" if object_name is None else "object",
        "object_name": object_name, "unavailable": {},
        "space_dimension": 3, "domain_count": 1, "boundary_count": 12,
        "entity_counts": VALUES["getNEntities"],
        "bounding_box": VALUES["getBoundingBox"], "finite_void_count": 1,
        "voids_are_labeled": True, "up_down_ext": VALUES["getUpDownExt"],
    }
    expected = ([call.exists()] if object_name else []) + [getattr(call, m)() for m in VALUES]
    assert info.mock_calls == expected
    if object_name:
        runtime.geom.obj.assert_called_once_with("fluid1")
        assert runtime.geom.mock_calls == [call.obj("fluid1"), call.obj().exists()] + [
            getattr(call.obj(), m)() for m in VALUES]
    else:
        runtime.obj.assert_not_called()
        assert runtime.obj.mock_calls == []
    runtime.require.assert_called_once_with("get_geometry_topology_info")


def test_unavailable_object_getter_no_fallback(runtime):
    del runtime.obj.getUpDownExt
    result = invoke("fluid1")
    assert result["success"]
    assert result["data"]["up_down_ext"] is None
    assert "unavailable" in result["data"]["unavailable"]["up_down_ext"]
    assert runtime.obj.mock_calls == [call.exists()] + [
        getattr(call, m)() for m in VALUES if m != "getUpDownExt"]
    runtime.geom.getUpDownExt.assert_not_called()


@pytest.mark.parametrize("object_name", [None, "fluid1"])
@pytest.mark.parametrize("method", list(VALUES))
def test_exception_stops_no_retry_or_mutation(runtime, object_name, method):
    info = runtime.geom if object_name is None else runtime.obj
    getattr(info, method).side_effect = RuntimeError("Original COMSOL API error")
    result = invoke(object_name)
    assert not result["success"]
    assert result["error"] == "Original COMSOL API error"
    methods = list(VALUES)
    assert info.mock_calls == ([call.exists()] if object_name else []) + [
        getattr(call, m)() for m in methods[:methods.index(method) + 1]]


def test_missing_object_no_creation(runtime):
    runtime.obj.exists.return_value = False
    result = invoke("fluid1")
    assert not result["success"]
    assert result["error"] == 'Geometry object "fluid1" does not exist in the current build state.'
    assert runtime.geom.mock_calls == [call.obj("fluid1"), call.obj().exists()]
    assert runtime.obj.mock_calls == [call.exists()]


def test_null_and_false_values_preserved(runtime):
    runtime.geom.getBoundingBox.return_value = None
    runtime.geom.voidsAreLabeled.return_value = False
    runtime.geom.getNFiniteVoids.return_value = 0
    result = invoke()["data"]
    assert result["bounding_box"] is None
    assert result["unavailable"] == {"bounding_box": "getBoundingBox returned null."}
    assert result["voids_are_labeled"] is False
    assert result["finite_void_count"] == 0


@pytest.mark.parametrize("kwargs", [{"component": ""}, {"geometry": " "}, {"object_name": ""}])
def test_invalid_identifiers_no_java_calls(runtime, kwargs):
    assert not invoke(**kwargs)["success"]
    assert runtime.model.java.mock_calls == []


def test_registration_classification_and_readonly_wrapper(runtime, monkeypatch):
    registry = Mock()
    tools.register(registry)
    assert call(tools.get_geometry_topology_info) in registry.add_tool.call_args_list
    assert "get_geometry_topology_info" in SAFE_READ_TOOLS
    read = Mock(wraps=tools._run_tool_readonly)
    write = Mock(side_effect=AssertionError("write wrapper forbidden"))
    monkeypatch.setattr(tools, "_run_tool_readonly", read)
    monkeypatch.setattr(tools, "_run_tool", write)
    assert invoke()["success"]
    read.assert_called_once()
    write.assert_not_called()
