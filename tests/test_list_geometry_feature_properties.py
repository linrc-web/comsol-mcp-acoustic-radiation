"""Mock-only enumeration: no JVM or COMSOL Server access."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import comsol_mcp._state as state
import comsol_mcp._tools_geometry as tools
from comsol_mcp._server import SAFE_READ_TOOLS


@pytest.fixture
def runtime(monkeypatch):
    model = Mock()
    geom = model.java.component.return_value.geom.return_value
    collection = Mock()
    collection.tags.return_value = ["feature1"]
    feature = Mock()
    feature.properties.return_value = ["property_a", "property_b", "property_c"]
    geom.feature.side_effect = lambda tag=None: collection if tag is None else feature
    require = Mock(return_value=model)
    monkeypatch.setattr(tools, "_require_visible_main", require)
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error)))
    return SimpleNamespace(model=model, geom=geom, collection=collection,
                           feature=feature, require=require)


def invoke(**kwargs):
    args = dict(component="comp1", geometry="geom1", feature_tag="feature1")
    args.update(kwargs)
    return json.loads(tools.list_geometry_feature_properties(**args))


def assert_readonly_calls(runtime):
    # Exhaustive call allowlist rejects create/set/remove/selection/run/build
    # at every Java object, including on exception paths.
    assert runtime.model.mock_calls == [
        call.java.component("comp1"), call.java.component().geom("geom1"),
        call.java.component().geom().feature(),
        call.java.component().geom().feature("feature1"),
    ]
    assert runtime.collection.mock_calls == [call.tags()]
    assert runtime.feature.mock_calls == [call.properties()]


@pytest.mark.parametrize("names", [["property_a", "property_b", "property_c"], []])
def test_enumeration_strictly_readonly(runtime, names):
    runtime.feature.properties.return_value = names
    result = invoke()
    assert result["success"]
    assert result["data"] == dict(component="comp1", geometry="geom1",
                                  feature_tag="feature1", property_names=names)
    assert_readonly_calls(runtime)


def test_missing_feature(runtime):
    runtime.collection.tags.return_value = []
    result = invoke()
    assert not result["success"]
    assert result["error"] == 'Geometry feature "feature1" does not exist.'
    assert runtime.geom.feature.call_args_list == [call()]
    assert runtime.feature.mock_calls == []


def test_java_error_no_retry_or_mutation(runtime):
    runtime.feature.properties.side_effect = RuntimeError("Original Java error")
    result = invoke()
    assert not result["success"]
    assert result["error"] == "Original Java error"
    assert_readonly_calls(runtime)


def test_null_is_not_empty_or_retried(runtime):
    runtime.feature.properties.return_value = None
    result = invoke()
    assert not result["success"]
    assert result["error"] == "Geometry feature properties() returned null."
    assert_readonly_calls(runtime)


@pytest.mark.parametrize("field", ["component", "geometry", "feature_tag"])
@pytest.mark.parametrize("value", ["", " ", None])
def test_invalid_identifiers_no_java_access(runtime, field, value):
    result = invoke(**{field: value})
    assert not result["success"]
    assert result["error"] == f"{field} is required."
    assert runtime.model.mock_calls == []


def test_readonly_wrapper_and_guard(runtime, monkeypatch):
    read = Mock(wraps=tools._run_tool_readonly)
    write = Mock(wraps=tools._run_tool)
    monkeypatch.setattr(tools, "_run_tool_readonly", read)
    monkeypatch.setattr(tools, "_run_tool", write)
    assert invoke()["success"]
    read.assert_called_once()
    write.assert_not_called()
    runtime.require.assert_called_once_with("list_geometry_feature_properties")


def test_registered_and_safe_read():
    registry = Mock()
    tools.register(registry)
    assert call(tools.list_geometry_feature_properties) in registry.add_tool.call_args_list
    assert "list_geometry_feature_properties" in SAFE_READ_TOOLS
