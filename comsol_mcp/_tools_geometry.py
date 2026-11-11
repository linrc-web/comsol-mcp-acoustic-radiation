#!/usr/bin/env python3
"""MCP tools: ensure_component, ensure_geometry, ensure_mesh, create/update/delete/run_feature."""

from __future__ import annotations

from typing import Any

from comsol_mcp._state import _run_tool, _run_tool_readonly
from comsol_mcp._model import _require_visible_main
from comsol_mcp._model_ops import (
    _normalize_properties, _ensure_geometry_java, _apply_feature_properties,
    _read_geometry_feature_properties,
    _list_geometry_feature_properties,
    _geometry_feature_object_names,
    _geometry_topology_info,
    _geometry_boundary_info,
)


def ensure_component(component: str = "comp1", dimension: int = 2) -> str:
    """Ensure a component exists in the selected server-side model."""

    def _impl() -> dict[str, Any]:
        from comsol_mcp._model_ops import _ensure_component_java
        return _ensure_component_java(_require_visible_main("ensure_component"), component.strip() or "comp1", int(dimension))

    return _run_tool("ensure_component", _impl)


def ensure_geometry(component: str = "comp1", geometry: str = "geom1", dimension: int = 2) -> str:
    """Ensure a geometry sequence exists in the selected server-side model."""

    def _impl() -> dict[str, Any]:
        return _ensure_geometry_java(
            _require_visible_main("ensure_geometry"),
            component.strip() or "comp1",
            geometry.strip() or "geom1",
            int(dimension),
        )

    return _run_tool("ensure_geometry", _impl)


def ensure_mesh(component: str = "comp1", mesh: str = "mesh1") -> str:
    """Ensure a mesh sequence exists in the selected server-side model."""

    def _impl() -> dict[str, Any]:
        from comsol_mcp._model_ops import _ensure_mesh_java
        return _ensure_mesh_java(_require_visible_main("ensure_mesh"), component.strip() or "comp1", mesh.strip() or "mesh1")

    return _run_tool("ensure_mesh", _impl)


def create_feature(
    component: str,
    geometry: str,
    tag: str,
    feature_type: str,
    properties_json: str = "[]",
    run_geometry: bool = False,
) -> str:
    """Create a geometry feature and optionally apply initial properties."""

    def _impl() -> dict[str, Any]:
        model = _require_visible_main("create_feature")
        comp = component.strip() or "comp1"
        geom = geometry.strip() or "geom1"
        feature_tag = tag.strip()
        kind = feature_type.strip()
        if not feature_tag:
            raise ValueError("Feature tag is required.")
        if not kind:
            raise ValueError("feature_type is required.")
        _ensure_geometry_java(model, comp, geom, 2)
        feature_container = model.java.component(comp).geom(geom).feature()
        if feature_tag not in list(feature_container.tags()):
            model.java.component(comp).geom(geom).create(feature_tag, kind)
        feature = model.java.component(comp).geom(geom).feature(feature_tag)
        applied = _apply_feature_properties(feature, _normalize_properties(properties_json))
        if run_geometry:
            model.java.component(comp).geom(geom).run()
        return {
            "component": comp,
            "geometry": geom,
            "tag": feature_tag,
            "feature_type": kind,
            "properties": applied,
            "run_geometry": bool(run_geometry),
        }

    return _run_tool("create_feature", _impl)


def update_feature(
    component: str,
    geometry: str,
    tag: str,
    properties_json: str,
    run_geometry: bool = False,
) -> str:
    """Update geometry feature properties on the selected server-side model."""

    def _impl() -> dict[str, Any]:
        model = _require_visible_main("update_feature")
        comp = component.strip() or "comp1"
        geom = geometry.strip() or "geom1"
        feature_tag = tag.strip()
        if not feature_tag:
            raise ValueError("Feature tag is required.")
        _ensure_geometry_java(model, comp, geom, 2)
        feature_container = model.java.component(comp).geom(geom).feature()
        if feature_tag not in list(feature_container.tags()):
            raise LookupError(f'Feature "{feature_tag}" does not exist.')
        feature = model.java.component(comp).geom(geom).feature(feature_tag)
        applied = _apply_feature_properties(feature, _normalize_properties(properties_json))
        if run_geometry:
            model.java.component(comp).geom(geom).run()
        return {
            "component": comp,
            "geometry": geom,
            "tag": feature_tag,
            "properties": applied,
            "run_geometry": bool(run_geometry),
        }

    return _run_tool("update_feature", _impl)


def delete_feature(component: str, geometry: str, tag: str, run_geometry: bool = False) -> str:
    """Delete a geometry feature from the selected server-side model."""

    def _impl() -> dict[str, Any]:
        model = _require_visible_main("delete_feature")
        comp = component.strip() or "comp1"
        geom = geometry.strip() or "geom1"
        feature_tag = tag.strip()
        if not feature_tag:
            raise ValueError("Feature tag is required.")
        feature_container = model.java.component(comp).geom(geom).feature()
        if feature_tag not in list(feature_container.tags()):
            raise LookupError(f'Feature "{feature_tag}" does not exist.')
        feature_container.remove(feature_tag)
        if run_geometry:
            model.java.component(comp).geom(geom).run()
        return {
            "component": comp,
            "geometry": geom,
            "tag": feature_tag,
            "run_geometry": bool(run_geometry),
        }

    return _run_tool("delete_feature", _impl)


def run_feature(collection: str, tag: str, component: str = "comp1") -> str:
    """Run a geometry or mesh sequence on the selected server-side model."""

    def _impl() -> dict[str, Any]:
        model = _require_visible_main("run_feature")
        comp = component.strip() or "comp1"
        seq_tag = tag.strip()
        kind = collection.strip().lower()
        if not seq_tag:
            raise ValueError("Feature tag is required.")
        if kind == "geometry":
            model.java.component(comp).geom(seq_tag).run()
        elif kind == "mesh":
            model.java.component(comp).mesh(seq_tag).run()
        else:
            raise ValueError('collection must be "geometry" or "mesh".')
        return {"collection": kind, "component": comp, "tag": seq_tag}

    return _run_tool("run_feature", _impl)


def _geometry_feature_selection(model, component, geometry, feature_tag, selection_name):
    """Look up an existing selection; never ensure/create geometry or features."""
    for name, value in (("component", component), ("geometry", geometry),
                        ("feature_tag", feature_tag), ("selection_name", selection_name)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required.")
    geom = model.java.component(component).geom(geometry)
    if feature_tag not in list(geom.feature().tags()):
        raise LookupError(f'Geometry feature "{feature_tag}" does not exist.')
    # COMSOL throws for an unknown selection name. Propagate without fallback.
    return geom.feature(feature_tag).selection(selection_name)


def set_geometry_feature_selection(
    component: str,
    geometry: str,
    feature_tag: str,
    selection_name: str,
    object_tags: list[str],
) -> str:
    """Set an existing geometry feature's object input selection; do not build.

    object_tags is an array of one or more geometry object tags, e.g.
    ["cuboid1"] for Rotate input or ["obj1", "obj2"] for Union input.
    Uses GeomObjectSelection.set(String... onames), not feature.set().
    """
    def _impl() -> dict[str, Any]:
        if (not isinstance(object_tags, list) or not object_tags
                or any(not isinstance(tag, str) or not tag.strip() for tag in object_tags)):
            raise ValueError("object_tags must be a non-empty array of non-empty strings.")
        model = _require_visible_main("set_geometry_feature_selection")
        selection = _geometry_feature_selection(
            model, component, geometry, feature_tag, selection_name)
        selection.set(object_tags)
        return {"component": component, "geometry": geometry, "feature_tag": feature_tag,
                "selection_name": selection_name, "object_tags": list(object_tags)}

    return _run_tool("set_geometry_feature_selection", _impl)


def get_geometry_feature_selection(
    component: str,
    geometry: str,
    feature_tag: str,
    selection_name: str,
) -> str:
    """Read existing geometry object tags via selection.objects(); never mutate.

    Does not create selections/features, initialize selections, or build geometry.
    """
    def _impl() -> dict[str, Any]:
        model = _require_visible_main("get_geometry_feature_selection")
        selection = _geometry_feature_selection(
            model, component, geometry, feature_tag, selection_name)
        objects = selection.objects()
        if objects is None:
            raise RuntimeError("Geometry selection.objects() returned null.")
        return {"component": component, "geometry": geometry, "feature_tag": feature_tag,
                "selection_name": selection_name, "object_tags": [str(tag) for tag in objects]}

    return _run_tool_readonly("get_geometry_feature_selection", _impl)


def get_geometry_feature_properties(
    component: str,
    geometry: str,
    feature_tag: str,
    property_names: list[str],
    include_string_representation: bool = False,
) -> str:
    """Read actual existing geometry properties using typed Java getters only.

    Returns name, value_type, getter, and value for each requested property.
    Optional string-array probes preserve typed values and report probe errors
    per property. String results do not guarantee original expression recovery.
    Does not create, set, remove, build, run, or change selections. Missing
    features/properties, unsupported types and Java errors fail without retry.
    """
    def _impl() -> dict[str, Any]:
        model = _require_visible_main("get_geometry_feature_properties")
        return _read_geometry_feature_properties(
            model, component, geometry, feature_tag, property_names,
            include_string_representation=include_string_representation)

    return _run_tool_readonly("get_geometry_feature_properties", _impl)


def list_geometry_feature_properties(
    component: str,
    geometry: str,
    feature_tag: str,
) -> str:
    """List actual property names of an existing geometry feature, read-only.

    Uses COMSOL PropFeature.properties(); never creates, sets, removes or builds.
    Java errors propagate without retries or alternative enumeration methods.
    """
    def _impl() -> dict[str, Any]:
        model = _require_visible_main("list_geometry_feature_properties")
        return _list_geometry_feature_properties(model, component, geometry, feature_tag)

    return _run_tool_readonly("list_geometry_feature_properties", _impl)


def get_geometry_feature_object_names(
    component: str,
    geometry: str,
    feature_tag: str,
) -> str:
    """Read existing feature outputs via COMSOL feature.objectNames().

    Returns names unchanged, including indices, or an empty array. Never
    creates, sets, removes, builds, or runs geometry. API errors propagate
    without retries; this tool does not build missing outputs.
    """
    def _impl() -> dict[str, Any]:
        model = _require_visible_main("get_geometry_feature_object_names")
        return _geometry_feature_object_names(model, component, geometry, feature_tag)

    return _run_tool_readonly("get_geometry_feature_object_names", _impl)


def get_geometry_topology_info(
    component: str,
    geometry: str,
    object_name: str | None = None,
) -> str:
    """Read built final/object topology using GeomInfo getters only.

    Never builds, measures, changes selections, or infers cavity boundaries.
    Missing getter bindings return null with an unavailable reason; API errors
    propagate without retries. Object entity numbering is local to the object.
    """
    def _impl() -> dict[str, Any]:
        model = _require_visible_main("get_geometry_topology_info")
        return _geometry_topology_info(model, component, geometry, object_name)

    return _run_tool_readonly("get_geometry_topology_info", _impl)


def get_geometry_boundary_info(
    component: str,
    geometry: str,
    boundary_ids: list[int],
) -> str:
    """Read final 3D geometry boundary adjacency and vertex coordinates only.

    IDs are 1-based and returned unchanged, in input order. Never creates
    selections, builds, measures, or infers boundary roles. Errors stop without retry.
    """
    def _impl() -> dict[str, Any]:
        model = _require_visible_main("get_geometry_boundary_info")
        return _geometry_boundary_info(model, component, geometry, boundary_ids)

    return _run_tool_readonly("get_geometry_boundary_info", _impl)


def register(mcp_instance) -> None:
    mcp_instance.add_tool(get_geometry_boundary_info)
    mcp_instance.add_tool(get_geometry_topology_info)
    mcp_instance.add_tool(get_geometry_feature_object_names)
    mcp_instance.add_tool(list_geometry_feature_properties)
    mcp_instance.add_tool(get_geometry_feature_properties)
    mcp_instance.add_tool(set_geometry_feature_selection)
    mcp_instance.add_tool(get_geometry_feature_selection)
    mcp_instance.add_tool(ensure_component)
    mcp_instance.add_tool(ensure_geometry)
    mcp_instance.add_tool(ensure_mesh)
    mcp_instance.add_tool(create_feature)
    mcp_instance.add_tool(update_feature)
    mcp_instance.add_tool(delete_feature)
    mcp_instance.add_tool(run_feature)
