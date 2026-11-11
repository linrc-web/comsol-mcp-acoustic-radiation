"""Geometry-bound physics creation, with no COMSOL connection or JVM."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from comsol_mcp import _physics_ops as ops
from comsol_mcp import _state, _tools_physics


def model_with(geometries=None):
    geometries = {"geom1": SimpleNamespace(getSDim=lambda: 3)} if geometries is None else geometries
    physics = Mock()
    physics.tags.return_value = []
    geom_list = SimpleNamespace(tags=lambda: list(geometries))
    comp = SimpleNamespace(
        geom=lambda tag=None: geom_list if tag is None else geometries[tag],
        physics=lambda tag=None: physics,
    )
    components = SimpleNamespace(tags=lambda: ["comp1"])
    model = SimpleNamespace(java=SimpleNamespace(
        component=lambda tag=None: components if tag is None else comp))
    return model, physics


def test_explicit_geometry_uses_exact_three_argument_overload():
    model, physics = model_with()
    result = ops._create_physics(model, "comp1", "acpr", "PressureAcoustics", geometry_tag="geom1")
    physics.create.assert_called_once_with("acpr", "PressureAcoustics", "geom1")
    assert result["created"] and result["dimension"] == 3


def test_geometry_dimension_overrides_workflow(monkeypatch):
    model, _ = model_with()
    monkeypatch.setattr(_state, "_read_workflow_state", lambda: {"model_dimension": 2})
    assert ops._auto_detect_dimension(model, "comp1") == 3


@pytest.mark.parametrize("geometry_tag", ["missing", " geom2 "])
def test_invalid_geometry_never_creates(geometry_tag):
    model, physics = model_with()
    with pytest.raises(ValueError, match="does not exist under component"):
        ops._create_physics(model, "comp1", "acpr", "PressureAcoustics", geometry_tag=geometry_tag)
    physics.create.assert_not_called()


def test_missing_component_never_creates():
    model, physics = model_with()
    with pytest.raises(ValueError, match='Component "missing" does not exist'):
        ops._create_physics(model, "missing", "acpr", "PressureAcoustics", geometry_tag="geom1")
    physics.create.assert_not_called()


@pytest.mark.parametrize("geometry_tag", [None, "", "  "])
def test_legacy_arguments_bind_unique_geometry_and_preserve_fields(geometry_tag):
    model, physics = model_with()
    ops._create_physics(model, "comp1", "acpr", "PressureAcoustics", 3, "p,q", geometry_tag)
    physics.create.assert_called_once_with("acpr", "PressureAcoustics", "geom1", ["p", "q"])


def test_unknown_dimension_still_binds_unique_geometry(monkeypatch):
    model, physics = model_with({"geom1": SimpleNamespace()})
    monkeypatch.setattr(_state, "_read_workflow_state", lambda: {})
    ops._create_physics(model, "comp1", "acpr", "PressureAcoustics")
    physics.create.assert_called_once_with("acpr", "PressureAcoustics", "geom1", ["u"])


def test_multiple_geometries_fail_without_guessing():
    model, physics = model_with({"geom1": SimpleNamespace(getSDim=lambda: 3),
                                 "geom2": SimpleNamespace(getSDim=lambda: 2)})
    with pytest.raises(ValueError, match="supply geometry_tag explicitly"):
        ops._create_physics(model, "comp1", "acpr", "PressureAcoustics")
    physics.create.assert_not_called()


def test_legitimate_zero_dimensional_call_preserved(monkeypatch):
    model, physics = model_with({})
    monkeypatch.setattr(_state, "_read_workflow_state", lambda: {})
    ops._create_physics(model, "comp1", "ge", "GlobalEquations")
    physics.create.assert_called_once_with("ge", "GlobalEquations")


def test_spatial_request_without_geometry_rejected(monkeypatch):
    model, physics = model_with({})
    monkeypatch.setattr(_state, "_read_workflow_state", lambda: {})
    with pytest.raises(ValueError, match="no 0D fallback"):
        ops._create_physics(model, "comp1", "acpr", "PressureAcoustics")
    physics.create.assert_not_called()


def test_existing_physics_is_not_recreated():
    model, physics = model_with()
    physics.tags.return_value = ["acpr"]
    assert ops._create_physics(model, "comp1", "acpr", "PressureAcoustics")["created"] is False
    physics.create.assert_not_called()


def test_tool_forwards_geometry_through_guard(monkeypatch):
    model, _ = model_with()
    guard = Mock(return_value=model)
    create = Mock(return_value={"created": True})
    monkeypatch.setattr(_tools_physics, "_require_visible_main", guard)
    monkeypatch.setattr(_tools_physics, "_create_physics", create)
    monkeypatch.setattr(_tools_physics, "_run_tool", lambda name, impl: impl())
    _tools_physics.create_physics("comp1", "acpr", "PressureAcoustics", geometry_tag="geom1")
    guard.assert_called_once_with("create_physics")
    create.assert_called_once_with(model, "comp1", "acpr", "PressureAcoustics", 0, "u", "geom1")
