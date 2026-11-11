"""Mock-only final boundary queries: no COMSOL initialization or connection."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import comsol_mcp._state as state
import comsol_mcp._tools_geometry as tools
from comsol_mcp._server import SAFE_READ_TOOLS


GETTERS = ["getNBoundaries", "getVertexCoord", "getBoundingBox", "getAdjExt", "getAdj"]


@pytest.fixture
def runtime(monkeypatch):
    # Every API outside this allowlist (including build/selection) fails immediately.
    g = Mock(spec_set=GETTERS)
    g.getNBoundaries.return_value = 12
    g.getVertexCoord.return_value = [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]]
    g.getBoundingBox.return_value = [10, 40, 11, 41, 12, 42]
    g.getAdjExt.side_effect = lambda a, b, idx: [-7, 1, 0] if idx == 12 else [1, -1]
    g.getAdj.side_effect = lambda a, b, idx: [4, 1] if idx == 12 else [2, 3]
    comp = Mock(spec_set=["geom"])
    comp.geom.return_value = g
    java = Mock(spec_set=["component"])
    java.component.return_value = comp
    require = Mock(return_value=SimpleNamespace(java=java))
    monkeypatch.setattr(tools, "_require_visible_main", require)
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error)))
    return SimpleNamespace(g=g, java=java, require=require)


def invoke(ids, component="comp1", geometry="geom1"):
    return json.loads(tools.get_geometry_boundary_info(component, geometry, ids))


def test_single_boundary_raw_ids_and_one_based_mapping(runtime):
    result = invoke([12])
    assert result["success"]
    assert result["data"] == {
        "component": "comp1", "geometry": "geom1", "space_dimension": 3,
        "geometry_bounding_box": [10, 40, 11, 41, 12, 42],
        "boundaries": [{"boundary_id": 12, "adjacent_domains_ext": [-7, 1, 0],
                        "vertex_ids": [4, 1],
                        "vertex_coordinates": [[40, 41, 42], [10, 11, 12]]}],
    }
    assert runtime.g.mock_calls == [call.getNBoundaries(), call.getVertexCoord(),
                                   call.getBoundingBox(), call.getAdjExt(2, 3, 12),
                                   call.getAdj(2, 0, 12)]
    runtime.require.assert_called_once_with("get_geometry_boundary_info")


def test_multiple_boundaries_preserve_both_input_and_vertex_order(runtime):
    data = invoke([12, 2, 1])["data"]
    assert [b["boundary_id"] for b in data["boundaries"]] == [12, 2, 1]
    assert data["boundaries"][1]["vertex_coordinates"] == [[20, 21, 22], [30, 31, 32]]
    assert runtime.g.getAdjExt.call_args_list == [call(2, 3, i) for i in [12, 2, 1]]
    assert runtime.g.getAdj.call_args_list == [call(2, 0, i) for i in [12, 2, 1]]
    runtime.g.getVertexCoord.assert_called_once()


@pytest.mark.parametrize("ids, message", [
    ([], "non-empty"), (None, "non-empty"), ((1,), "non-empty"),
    ([1.0], "integers only"), (["1"], "integers only"), ([True], "integers only"),
    ([0], "positive"), ([-1], "positive"), ([1, 1], "duplicate"),
])
def test_invalid_ids_before_java(runtime, ids, message):
    result = invoke(ids)
    assert not result["success"]
    assert message in result["error"]
    assert runtime.java.mock_calls == []


def test_out_of_range_validates_entire_request_before_query(runtime):
    result = invoke([1, 13])
    assert not result["success"]
    assert "13 exceeds boundary_count 12" in result["error"]
    assert runtime.g.mock_calls == [call.getNBoundaries()]


@pytest.mark.parametrize("component, geometry", [("", "geom1"), ("comp1", " ")])
def test_invalid_names(runtime, component, geometry):
    assert not invoke([1], component, geometry)["success"]
    assert runtime.java.mock_calls == []


@pytest.mark.parametrize("method", GETTERS)
def test_getter_exception_stops_without_retry(runtime, method):
    getattr(runtime.g, method).side_effect = RuntimeError("Original COMSOL getter error")
    result = invoke([12, 1])
    assert not result["success"]
    assert result["error"] == "Original COMSOL getter error"
    expected = [call.getNBoundaries(), call.getVertexCoord(), call.getBoundingBox(),
                call.getAdjExt(2, 3, 12), call.getAdj(2, 0, 12)]
    assert runtime.g.mock_calls == expected[:GETTERS.index(method) + 1]


@pytest.mark.parametrize("coordinates", [[[1], [2]], [[1, 2], [3], [4, 5]]])
def test_bad_coordinate_layout_fails_without_adjacency(runtime, coordinates):
    runtime.g.getVertexCoord.return_value = coordinates
    assert not invoke([1])["success"]
    runtime.g.getAdjExt.assert_not_called()


@pytest.mark.parametrize("vertex_ids", [[0], [-1], [5]])
def test_bad_returned_vertex_id_is_not_silently_indexed(runtime, vertex_ids):
    runtime.g.getAdj.side_effect = None
    runtime.g.getAdj.return_value = vertex_ids
    assert not invoke([1])["success"]


def test_registration_safe_read_and_readonly_wrapper(runtime, monkeypatch):
    registry = Mock()
    tools.register(registry)
    assert call(tools.get_geometry_boundary_info) in registry.add_tool.call_args_list
    assert "get_geometry_boundary_info" in SAFE_READ_TOOLS
    read = Mock(wraps=tools._run_tool_readonly)
    write = Mock(side_effect=AssertionError("write wrapper forbidden"))
    monkeypatch.setattr(tools, "_run_tool_readonly", read)
    monkeypatch.setattr(tools, "_run_tool", write)
    assert invoke([1])["success"]
    read.assert_called_once()
    write.assert_not_called()
