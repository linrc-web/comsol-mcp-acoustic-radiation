"""Physics feature creation tests using mocks only; no COMSOL or JVM startup."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import comsol_mcp._physics_ops as ops
import comsol_mcp._state as state
import comsol_mcp._tools_physics as tools


@pytest.fixture
def runtime(monkeypatch):
    feature = Mock()
    collection = Mock()
    collection.tags.return_value = []
    physics = Mock()
    physics.feature.side_effect = lambda tag=None: collection if tag is None else feature
    model = Mock()
    model.java.component.return_value.physics.return_value = physics
    monkeypatch.setattr(tools, "_require_visible_main", lambda tool: model)
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error)))
    # Avoid importing JPype: verify conversion is requested, without starting a JVM.
    java_int = Mock(side_effect=lambda value: value)
    monkeypatch.setattr(ops, "_java_int", java_int)
    return SimpleNamespace(model=model, feature=feature, collection=collection, java_int=java_int)


def call(**kwargs):
    args = dict(component="comp1", physics_tag="acpr", feature_tag="nvel1",
                feature_type="NormalVelocity")
    args.update(kwargs)
    return json.loads(tools.create_physics_feature(**args))


def test_omitted_dimension_preserves_old_call(runtime):
    # Existing positional properties_json argument must keep its meaning.
    result = json.loads(tools.create_physics_feature(
        "comp1", "acpr", "nvel1", "NormalVelocity", '{"nvel":"1e-3[m/s]"}'))
    assert result["success"] and result["data"]["created"]
    runtime.collection.create.assert_called_once_with("nvel1", "NormalVelocity")
    runtime.java_int.assert_not_called()
    runtime.feature.set.assert_called_once_with("nvel", "1e-3[m/s]")


@pytest.mark.parametrize("dimension", [0, 1, 2, 3])
def test_explicit_dimension_uses_three_arguments(runtime, dimension):
    result = call(dimension=dimension)
    assert result["success"] and result["data"]["created"]
    runtime.collection.create.assert_called_once_with("nvel1", "NormalVelocity", dimension)
    runtime.java_int.assert_called_once_with(dimension)


@pytest.mark.parametrize("dimension", [-1, 4, 2.0, "2", True, False, [], {}])
def test_invalid_dimension_rejected_before_java(runtime, dimension):
    result = call(dimension=dimension)
    assert not result["success"]
    assert "dimension must be an integer between 0 and 3" in result["error"]
    runtime.model.java.component.assert_not_called()
    runtime.collection.create.assert_not_called()


@pytest.mark.parametrize("field", ["physics_tag", "feature_tag", "feature_type"])
def test_required_inputs_still_validated(runtime, field):
    result = call(**{field: " ", "dimension": 2})
    assert not result["success"] and "required" in result["error"]
    runtime.collection.create.assert_not_called()


@pytest.mark.parametrize("dimension", [None, 2])
def test_java_creation_failure_is_reported_without_retry(runtime, dimension):
    runtime.collection.create.side_effect = RuntimeError("Java creation failed")
    result = call(dimension=dimension, properties_json='{"nvel":"1e-3[m/s]"}')
    assert not result["success"] and result["error"] == "Java creation failed"
    assert runtime.collection.create.call_count == 1
    runtime.feature.set.assert_not_called()


def test_property_failure_is_reported_without_retry(runtime):
    runtime.feature.set.side_effect = RuntimeError("Java property failed")
    result = call(dimension=2, properties_json='{"nvel":"1e-3[m/s]"}')
    assert not result["success"] and result["error"] == "Java property failed"
    runtime.collection.create.assert_called_once_with("nvel1", "NormalVelocity", 2)
    assert runtime.feature.set.call_count == 1


def test_existing_feature_not_recreated(runtime):
    runtime.collection.tags.return_value = ["nvel1"]
    result = call(dimension=2)
    assert result["success"] and result["data"]["created"] is False
    runtime.collection.create.assert_not_called()
