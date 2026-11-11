"""Selection getter tests: mock only, no JVM or live Server."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from comsol_mcp._physics_ops import _get_physics_selection
import comsol_mcp._tools_physics as tools
import comsol_mcp._state as state
from comsol_mcp._server import SAFE_READ_TOOLS, SAFE_VISIBLE_MAIN_WRITE_TOOLS


@pytest.fixture
def runtime(monkeypatch):
    selection = Mock(name="selection", spec=["dim", "entities"])
    selection.dim.return_value = 3
    selection.entities.return_value = [12, 1, 5]
    physics = Mock()
    feature = Mock()
    physics.selection.return_value = selection
    feature.selection.return_value = selection
    features = Mock()
    features.tags.return_value = ["shb1"]
    physics.feature.side_effect = lambda tag=None: features if tag is None else feature
    collection = Mock()
    collection.tags.return_value = ["acpr"]
    comp = Mock()
    comp.physics.side_effect = lambda tag=None: collection if tag is None else physics
    model = Mock()
    model.java.component.return_value = comp
    monkeypatch.setattr(tools, "_require_visible_main", Mock(return_value=model))
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error)))
    return SimpleNamespace(**locals())


def invoke(feature_tag=None):
    return json.loads(tools.get_physics_selection("comp1", "acpr", feature_tag))


@pytest.mark.parametrize("ftag", [None, "shb1"])
def test_scope_and_no_mutation(runtime, ftag):
    result = invoke(ftag)
    assert result["success"]
    data = result["data"]
    assert data["scope"] == ("physics" if ftag is None else "feature")
    assert data["feature_tag"] == ftag
    assert data["entity_ids"] == [12, 1, 5]  # preserve order
    assert data["geometry_dimension"] == 3
    assert data["selection_all"] is None
    assert "selection_all" in data["unavailable"]
    assert runtime.selection.mock_calls == [call.dim(), call.entities()]
    assert runtime.model.mock_calls == [call.java.component("comp1"),
                                        call.java.component().physics(),
                                        call.java.component().physics("acpr")]
    assert runtime.comp.mock_calls == [call.physics(), call.physics("acpr")]
    assert runtime.collection.mock_calls == [call.tags()]
    assert runtime.physics.mock_calls == (
        [call.selection()]
        if ftag is None else [call.feature(), call.feature("shb1")])
    if ftag is not None:
        assert runtime.features.mock_calls == [call.tags()]
        assert runtime.feature.mock_calls == [call.selection()]


@pytest.mark.parametrize("entities", [[1], [], None])
def test_all_state_cannot_be_inferred(runtime, entities):
    # All and explicit selections may return identical lists or null.
    runtime.selection.entities.return_value = entities
    data = invoke()["data"]
    assert data["selection_all"] is None
    assert data["entity_ids"] == entities
    assert ("entity_ids" in data["unavailable"]) == (entities is None)


@pytest.mark.parametrize("missing,field", [("dim", "geometry_dimension"), ("entities", "entity_ids")])
def test_unavailable_binding(runtime, missing, field):
    delattr(runtime.selection, missing)
    data = invoke()["data"]
    assert data[field] is None
    assert "unavailable" in data["unavailable"][field]


@pytest.mark.parametrize("method", ["dim", "entities"])
def test_getter_exception_stops_without_retry(runtime, method):
    getattr(runtime.selection, method).side_effect = RuntimeError("original Java error")
    result = invoke()
    assert not result["success"]
    assert result["error"] == "original Java error"
    assert getattr(runtime.selection, method).call_count == 1
    if method == "dim":
        runtime.selection.entities.assert_not_called()


def test_missing_physics(runtime):
    runtime.collection.tags.return_value = []
    result = invoke()
    assert not result["success"] and 'Physics "acpr" does not exist' in result["error"]
    runtime.physics.selection.assert_not_called()


def test_missing_feature(runtime):
    runtime.features.tags.return_value = []
    result = invoke("shb1")
    assert not result["success"] and 'feature "shb1" does not exist' in result["error"]
    runtime.feature.selection.assert_not_called()


def test_selection_access_exception(runtime):
    runtime.physics.selection.side_effect = RuntimeError("selection unavailable")
    assert invoke()["error"] == "selection unavailable"
    runtime.physics.selection.assert_called_once_with()


def test_readonly_wrapper_and_classification(runtime, monkeypatch):
    read = Mock(wraps=tools._run_tool_readonly)
    write = Mock(side_effect=AssertionError("mutation wrapper"))
    monkeypatch.setattr(tools, "_run_tool_readonly", read)
    monkeypatch.setattr(tools, "_run_tool", write)
    assert invoke()["success"]
    read.assert_called_once()
    write.assert_not_called()
    assert "get_physics_selection" in SAFE_READ_TOOLS
    assert "get_physics_selection" not in SAFE_VISIBLE_MAIN_WRITE_TOOLS


@pytest.mark.parametrize("args", [("", "acpr", None), ("comp1", "", None), ("comp1", "acpr", " ")])
def test_invalid_tags_before_model_access(runtime, args):
    assert not json.loads(tools.get_physics_selection(*args))["success"]
    runtime.model.java.component.assert_not_called()
