"""Mesh-only tools. No implicit sequence creation, builds, solves or retries."""
from __future__ import annotations

from comsol_mcp._state import _run_tool, _run_tool_readonly
from comsol_mcp._model import _require_model, _require_visible_main
from comsol_mcp._model_ops import (
    _mesh_sequence, _mesh_feature, _normalize_properties, _apply_feature_properties,
    _read_mesh_feature_property_values,
)


def _required(**tags):
    for name, value in tags.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required.")
    return {name: value.strip() for name, value in tags.items()}


def list_mesh_features(component: str, mesh: str) -> str:
    """List sequence-level mesh features in API order, without building or ensuring.

    Uses MeshFeatureList.tags(), getType() and the zero-argument label() getter.
    Index is zero-based; no sorting or inference from tags is performed.
    Missing getters or Java errors fail explicitly without mutation or retry.
    """
    def impl():
        tags = _required(component=component, mesh=mesh)
        seq = _mesh_sequence(_require_visible_main("list_mesh_features"), **tags)
        collection = seq.feature()
        getter = getattr(collection, "tags", None)
        if not callable(getter):
            raise RuntimeError("Mesh feature tags() getter unavailable.")
        feature_tags = getter()
        if feature_tags is None:
            raise RuntimeError("Mesh feature tags() returned null.")
        features = []
        for index, tag in enumerate(feature_tags):
            tag = str(tag)
            feature = seq.feature(tag)
            row = {"index": index, "tag": tag}
            for field, method in (("type", "getType"), ("label", "label")):
                getter = getattr(feature, method, None)
                if not callable(getter):
                    raise RuntimeError(f'Mesh feature "{tag}" {method}() getter unavailable.')
                value = getter()
                if value is None:
                    raise RuntimeError(f'Mesh feature "{tag}" {method}() returned null.')
                row[field] = str(value)
            features.append(row)
        return {**tags, "features": features}
    return _run_tool_readonly("list_mesh_features", impl)


def create_mesh_feature(component: str, mesh: str, feature_tag: str, feature_type: str) -> str:
    """Create a mesh feature, e.g. FreeTet or Size, using create(tag, type).

    Existing tags fail without overwrite. Dimension belongs to selection,
    not mesh feature creation. Never ensures a sequence or builds the mesh.
    """
    def impl():
        tags = _required(component=component, mesh=mesh, feature_tag=feature_tag)
        kind = _required(feature_type=feature_type)["feature_type"]
        seq = _mesh_sequence(_require_visible_main("create_mesh_feature"), tags["component"], tags["mesh"])
        if tags["feature_tag"] in [str(t) for t in seq.feature().tags()]:
            raise ValueError(f'Mesh feature "{tags["feature_tag"]}" already exists.')
        seq.create(tags["feature_tag"], kind)
        return {**tags, "feature_type": kind, "created": True}
    return _run_tool("create_mesh_feature", impl)


def update_mesh_feature(component: str, mesh: str, feature_tag: str, properties_json: str) -> str:
    """Set mesh properties from a JSON object or name/value(s) array.

    Uses the existing scalar-string / string-array property validation.
    Does not whitelist property names, build, or claim independent readback.
    """
    def impl():
        tags = _required(component=component, mesh=mesh, feature_tag=feature_tag)
        props = _normalize_properties(properties_json)
        feature = _mesh_feature(_require_visible_main("update_mesh_feature"), **tags)
        return {**tags, "properties_applied": _apply_feature_properties(feature, props)}
    return _run_tool("update_mesh_feature", impl)


def set_mesh_feature_selection(component: str, mesh: str, feature_tag: str,
                               dimension: int, entity_ids: list[int]) -> str:
    """Set explicit mesh feature selection using selection().geom(dim), set(ids).

    Supports entity dimensions 0..3, including 3D domains (3) and boundaries (2).
    Empty IDs explicitly set an empty selection; never calls all().
    """
    def impl():
        tags = _required(component=component, mesh=mesh, feature_tag=feature_tag)
        if type(dimension) is not int or dimension not in range(4):
            raise ValueError("dimension must be an integer from 0 to 3.")
        if not isinstance(entity_ids, list) or any(type(i) is not int or i <= 0 for i in entity_ids):
            raise ValueError("entity_ids must be a list of positive integer IDs.")
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("entity_ids must not contain duplicates.")
        feature = _mesh_feature(_require_visible_main("set_mesh_feature_selection"), **tags)
        selection = feature.selection()
        selection.geom(dimension)
        selection.set(entity_ids)
        return {**tags, "geometry_dimension": dimension, "entity_ids": entity_ids, "selection_set": True}
    return _run_tool("set_mesh_feature_selection", impl)


def get_mesh_feature_properties(component: str, mesh: str, feature_tag: str) -> str:
    """Strict read-only metadata: actual type, label and property names, not values."""
    def impl():
        tags = _required(component=component, mesh=mesh, feature_tag=feature_tag)
        feature = _mesh_feature(_require_visible_main("get_mesh_feature_properties"), **tags)
        return {**tags, "tag": tags["feature_tag"], "feature_type": str(feature.getType()),
                "label": str(feature.label()), "property_names": [str(p) for p in feature.properties()]}
    return _run_tool_readonly("get_mesh_feature_properties", impl)


def get_mesh_feature_property_values(component: str, mesh: str, feature_tag: str,
                                     property_names: list[str]) -> str:
    """Strict read-only typed values, in requested order; no evaluation or retry.

    Uses hasProperty/getValueType and one documented typed getter per property.
    Unreadable properties have available=false and an error; empty input returns
    an empty list after validating the existing component/mesh/feature.
    String values are preserved verbatim, including expressions.
    """
    def impl():
        tags = _required(component=component, mesh=mesh, feature_tag=feature_tag)
        feature = _mesh_feature(_require_visible_main("get_mesh_feature_property_values"), **tags)
        return {**tags, "properties": _read_mesh_feature_property_values(feature, property_names)}
    return _run_tool_readonly("get_mesh_feature_property_values", impl)


def get_mesh_feature_selection(component: str, mesh: str, feature_tag: str) -> str:
    """Strict read-only dim()/entities(); preserves API order and empty selections.

    No all-selection inference. Missing bindings/null IDs are unavailable;
    Java getter errors propagate without retries.
    """
    def impl():
        tags = _required(component=component, mesh=mesh, feature_tag=feature_tag)
        feature = _mesh_feature(_require_visible_main("get_mesh_feature_selection"), **tags)
        sel = feature.selection()
        data = {**tags, "geometry_dimension": None, "entity_ids": None, "unavailable": {}}
        for field, method in (("geometry_dimension", "dim"), ("entity_ids", "entities")):
            getter = getattr(sel, method, None)
            if getter is None:
                data["unavailable"][field] = f"{method} getter unavailable."
                continue
            value = getter()
            if value is None:
                data["unavailable"][field] = f"{method} returned null."
            else:
                data[field] = int(value) if field == "geometry_dimension" else [int(i) for i in value]
        return data
    return _run_tool_readonly("get_mesh_feature_selection", impl)


def set_mesh_automatic(component: str, mesh: str, automatic: bool) -> str:
    """Explicitly set mesh control mode; never builds or ensures a sequence."""
    def impl():
        tags = _required(component=component, mesh=mesh)
        if type(automatic) is not bool:
            raise ValueError("automatic must be a bool.")
        seq = _mesh_sequence(_require_visible_main("set_mesh_automatic"), **tags)
        seq.automatic(automatic)
        return {**tags, "automatic": automatic, "setter_accepted": True}
    return _run_tool("set_mesh_automatic", impl)


def _mesh_activity(feature):
    """Read activity once, preserving unavailable/error states without retry."""
    try:
        getter = getattr(feature, "isActive", None)
        if not callable(getter):
            raise RuntimeError("isActive() getter unavailable.")
        value = getter()
        if value is None:
            raise RuntimeError("isActive() returned null.")
        return bool(value), True, None
    except Exception as exc:
        return None, False, str(exc)


def get_mesh_feature_active(component: str, mesh: str, feature_tag: str) -> str:
    """Read existing feature activity only; never build or change the model."""
    def impl():
        tags = _required(component=component, mesh=mesh, feature_tag=feature_tag)
        feature = _mesh_feature(_require_model(), **tags)
        active, available, error = _mesh_activity(feature)
        return {**tags, "active": active, "available": available, "error": error}
    return _run_tool_readonly("get_mesh_feature_active", impl)


def set_mesh_feature_active(component: str, mesh: str, feature_tag: str, active: bool) -> str:
    """Set only feature activity under the visible-main guard; never build/save."""
    def impl():
        tags = _required(component=component, mesh=mesh, feature_tag=feature_tag)
        if type(active) is not bool:
            raise ValueError("active must be a bool.")
        feature = _mesh_feature(_require_visible_main("set_mesh_feature_active"), **tags)
        feature.active(active)
        resulting, available, error = _mesh_activity(feature)
        if available and resulting != active:
            error = "Activity readback does not match requested state."
        return {**tags, "requested_active": active, "resulting_active": resulting,
                "available": available, "success": available and resulting == active,
                "error": error}
    return _run_tool("set_mesh_feature_active", impl)


def move_mesh_feature(component: str, mesh: str, feature_tag: str, position: int) -> str:
    """Move an existing mesh feature to a zero-based COMSOL destination index.

    Requires the visible-main mutation guard. Only changes sequence order;
    never sets properties, creates features, builds, or saves. Reads tags once
    after moving to verify the destination, without retrying the mutation.
    """
    def impl():
        tags = _required(component=component, mesh=mesh, feature_tag=feature_tag)
        if type(position) is not int or position < 0:
            raise ValueError("position must be a non-negative integer.")
        seq = _mesh_sequence(_require_visible_main("move_mesh_feature"),
                             tags["component"], tags["mesh"])
        collection = seq.feature()
        before = [str(tag) for tag in collection.tags()]
        if tags["feature_tag"] not in before:
            raise ValueError(f'Mesh feature "{tags["feature_tag"]}" does not exist.')
        if position >= len(before):
            raise ValueError(f"position must be less than the feature count ({len(before)}).")
        collection.move(tags["feature_tag"], position)
        data = {**tags, "requested_position": position, "resulting_position": None,
                "ordered_features": None, "success": False, "error": None}
        try:
            after = collection.tags()
            if after is None:
                raise RuntimeError("Mesh feature tags() returned null after move.")
            ordered = [str(tag) for tag in after]
            data["ordered_features"] = ordered
            if tags["feature_tag"] in ordered:
                data["resulting_position"] = ordered.index(tags["feature_tag"])
            expected = before.copy()
            expected.remove(tags["feature_tag"])
            expected.insert(position, tags["feature_tag"])
            data["success"] = ordered == expected
            if not data["success"]:
                data["error"] = "Mesh feature order readback does not match requested move."
        except Exception as exc:
            data["error"] = str(exc)
        return data
    return _run_tool("move_mesh_feature", impl)


def register(mcp_instance) -> None:
    for tool in (list_mesh_features, create_mesh_feature, update_mesh_feature, set_mesh_feature_selection,
                 get_mesh_feature_properties, get_mesh_feature_property_values,
                 get_mesh_feature_selection, set_mesh_automatic,
                 get_mesh_feature_active, set_mesh_feature_active, move_mesh_feature):
        mcp_instance.add_tool(tool)
