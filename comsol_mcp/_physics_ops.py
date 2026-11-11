#!/usr/bin/env python3
"""Pure helpers for physics interface and variable operations (no global state)."""

from __future__ import annotations

from typing import Any

from comsol_mcp._model_ops import _normalize_properties, _apply_feature_properties


# ---------------------------------------------------------------------------
# Physics interface helpers
# ---------------------------------------------------------------------------
def _get_physics_selection(
    model: Any, component: str, physics_tag: str, feature_tag: str | None = None,
) -> dict[str, Any]:
    """Read documented COMSOL 6.4 selection getters; never initialize selections.

    Selection/LocalSelection/XDLocalSelection do not document isAll().
    all() is a setter; isGeom() means whole geometry, not the all-entities flag.
    https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/api/com/comsol/model/Selection.html
    https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/api/com/comsol/model/AbstractSelection.html
    """
    comp = model.java.component(component)
    if physics_tag not in [str(tag) for tag in comp.physics().tags()]:
        raise ValueError(f'Physics "{physics_tag}" does not exist.')
    target = comp.physics(physics_tag)
    if feature_tag is not None:
        if feature_tag not in [str(tag) for tag in target.feature().tags()]:
            raise ValueError(f'Physics feature "{feature_tag}" does not exist.')
        target = target.feature(feature_tag)
    selection = target.selection()
    unavailable = {
        "selection_all": "COMSOL 6.4 documents no all-entities flag getter; "
                         "all() mutates and isGeom() is not equivalent."
    }
    result: dict[str, Any] = {
        "component": component, "physics_tag": physics_tag,
        "scope": "physics" if feature_tag is None else "feature",
        "feature_tag": feature_tag, "selection_all": None,
        "geometry_dimension": None, "entity_ids": None,
        "unavailable": unavailable,
    }
    for field, method_name, convert in (
        ("geometry_dimension", "dim", int),
        ("entity_ids", "entities", lambda values: [int(v) for v in values]),
    ):
        # Only missing bindings are unavailable. Getter exceptions propagate;
        # do not retry, use alternative APIs, or modify selection to recover.
        try:
            getter = getattr(selection, method_name)
        except AttributeError:
            unavailable[field] = f"{method_name}() getter unavailable."
            continue
        value = getter()
        if value is None:
            unavailable[field] = f"{method_name}() returned null."
        else:
            result[field] = convert(value)
    return result


def _list_physics(model: Any, component: str) -> list[dict[str, Any]]:
    comp = model.java.component(component)
    tags = list(comp.physics().tags())
    result = []
    for tag in tags:
        phys = comp.physics(tag)
        info: dict[str, Any] = {"tag": tag}
        try:
            info["label"] = str(phys.label())
        except Exception:
            pass
        try:
            features = list(phys.feature().tags())
            info["features"] = features
            info["feature_count"] = len(features)
        except Exception:
            info["features"] = []
            info["feature_count"] = 0
        result.append(info)
    return result


def _list_physics_features(model: Any, component: str, physics_tag: str) -> list[dict[str, Any]]:
    phys = model.java.component(component).physics(physics_tag)
    tags = list(phys.feature().tags())
    result = []
    for tag in tags:
        feat = phys.feature(tag)
        info: dict[str, Any] = {"tag": tag}
        try:
            info["label"] = str(feat.label())
        except Exception:
            pass
        try:
            info["feature_type"] = str(feat.identifier())
        except Exception:
            # identifier() may not exist in all COMSOL versions; try alternatives
            try:
                info["feature_type"] = str(feat.getType())
            except Exception:
                try:
                    info["feature_type"] = str(feat.getString("type"))
                except Exception:
                    pass
        try:
            props = list(feat.properties())
            info["properties"] = props
        except Exception:
            info["properties"] = []
        # Check selection editability
        try:
            sel = feat.selection()
            info["selection_editable"] = not getattr(sel, 'isInherited', lambda: False)()
        except Exception:
            pass
        result.append(info)
    return result


def _auto_detect_dimension(model: Any, component: str) -> int:
    """Prefer actual geometry dimensions, then workflow and physics fallbacks."""
    comp = model.java.component(component)
    for geom_tag in comp.geom().tags():
        dim = _geometry_dimension(comp.geom(geom_tag))
        if dim:
            return dim
    # 1. Check workflow state
    try:
        from comsol_mcp._state import _read_workflow_state
        state = _read_workflow_state()
        dim = int(state.get("model_dimension", 0))
        if dim in (1, 2, 3):
            return dim
    except Exception:
        pass

    # 2. Try to detect from existing physics interfaces
    try:
        comp = model.java.component(component)
        phys_tags = list(comp.physics().tags())
        if phys_tags:
            phys = comp.physics(phys_tags[0])
            for method_name in ("getNDim", "getSpatialDim", "sdim", "getDim"):
                try:
                    val = getattr(phys, method_name)
                    if callable(val):
                        d = int(val())
                        if d in (1, 2, 3):
                            return d
                    else:
                        d = int(val)
                        if d in (1, 2, 3):
                            return d
                except Exception:
                    continue
    except Exception:
        pass

    # 3. Try geometry API methods
    try:
        comp = model.java.component(component)
        geom_tags = list(comp.geom().tags())
        if geom_tags:
            geom = comp.geom(geom_tags[0])
            for method_name in ("dimension", "getDimension", "dim", "spatialDim"):
                try:
                    val = getattr(geom, method_name)
                    if callable(val):
                        return int(val())
                    return int(val)
                except Exception:
                    continue
    except Exception:
        pass

    return 0


def _geometry_dimension(geom: Any) -> int:
    """Read geometry dimensions without building or changing the geometry."""
    for name in ("getSDim", "dimension", "getDimension", "dim", "spatialDim"):
        try:
            value = getattr(geom, name)
            dim = int(value() if callable(value) else value)
            if dim in (1, 2, 3):
                return dim
        except Exception:
            continue
    return 0


def _java_int(value: int):
    """Convert Python int to Java int for COMSOL API overload resolution."""
    try:
        import jpype
        return jpype.JInt(value)
    except ImportError:
        return value


def _create_physics(model: Any, component: str, tag: str, physics_type: str, dimension: int = 0, dependent_variables: str = "u", geometry_tag: str | None = None) -> dict[str, Any]:
    if component not in [str(t) for t in model.java.component().tags()]:
        raise ValueError(f'Component "{component}" does not exist.')
    comp = model.java.component(component)
    geom_tags = [str(t) for t in comp.geom().tags()]
    if geometry_tag is not None and not isinstance(geometry_tag, str):
        raise ValueError("geometry_tag must be a string or None.")
    requested_geometry = geometry_tag.strip() if geometry_tag is not None else ""
    if requested_geometry and requested_geometry not in geom_tags:
        raise ValueError(f'Geometry "{requested_geometry}" does not exist under component "{component}".')
    existing = list(comp.physics().tags())
    if tag in existing:
        return {"tag": tag, "physics_type": physics_type, "created": False, "message": "Physics already exists."}
    dim = int(dimension)
    if dim <= 0:
        dim = _auto_detect_dimension(model, component)

    if requested_geometry:
        comp.physics().create(tag, physics_type, requested_geometry)
        dim = _geometry_dimension(comp.geom(requested_geometry))
    elif geom_tags:
        if len(geom_tags) != 1:
            raise ValueError("Multiple geometries exist; supply geometry_tag explicitly. No physics was created.")
        # A unique geometry establishes association even if dimension getters fail.
        geometry_tag = geom_tags[0]
        dim = _geometry_dimension(comp.geom(geometry_tag)) or dim
        dep_vars = [v.strip() for v in dependent_variables.split(",") if v.strip()]
        if not dep_vars:
            dep_vars = ["u"]
        comp.physics().create(tag, physics_type, geometry_tag, dep_vars)
    elif dim > 0 or physics_type == "PressureAcoustics":
        raise ValueError("Spatial physics requires an existing geometry; no 0D fallback was attempted.")
    else:
        comp.physics().create(tag, physics_type)
    return {"tag": tag, "physics_type": physics_type, "dimension": dim, "created": True}


def _remove_physics(model: Any, component: str, tag: str) -> dict[str, Any]:
    comp = model.java.component(component)
    existing = list(comp.physics().tags())
    if tag not in existing:
        raise LookupError(f'Physics "{tag}" does not exist in {component}.')
    comp.physics().remove(tag)
    return {"tag": tag, "removed": True}


def _create_physics_feature(
    model: Any, component: str, physics_tag: str,
    feature_tag: str, feature_type: str, dimension: int | None = None,
) -> dict[str, Any]:
    if dimension is not None and (type(dimension) is not int or not 0 <= dimension <= 3):
        raise ValueError("dimension must be an integer between 0 and 3, or None.")
    phys = model.java.component(component).physics(physics_tag)
    existing = list(phys.feature().tags())
    if feature_tag in existing:
        return {"tag": feature_tag, "feature_type": feature_type, "created": False, "message": "Feature already exists."}
    if dimension is None:
        phys.feature().create(feature_tag, feature_type)
    else:
        phys.feature().create(feature_tag, feature_type, _java_int(dimension))
    return {"tag": feature_tag, "feature_type": feature_type, "created": True}


def _remove_physics_feature(model: Any, component: str, physics_tag: str, feature_tag: str) -> dict[str, Any]:
    phys = model.java.component(component).physics(physics_tag)
    existing = list(phys.feature().tags())
    if feature_tag not in existing:
        raise LookupError(f'Feature "{feature_tag}" does not exist under physics "{physics_tag}".')
    phys.feature().remove(feature_tag)
    return {"tag": feature_tag, "removed": True}


def _apply_physics_properties(
    model: Any, component: str, physics_tag: str, feature_tag: str,
    properties_json: str,
) -> list[dict[str, Any]]:
    feat = model.java.component(component).physics(physics_tag).feature(feature_tag)
    props = _normalize_properties(properties_json)
    return _apply_feature_properties(feat, props)


def _set_physics_selection(
    model: Any, component: str, physics_tag: str, feature_tag: str,
    entities: list[int],
) -> dict[str, Any]:
    phys = model.java.component(component).physics(physics_tag)
    feat = phys.feature(feature_tag)
    try:
        sel = feat.selection()
        # Check if selection is inherited (not directly editable)
        try:
            if sel.isInherited():
                raise ValueError(
                    f'Feature "{feature_tag}" has an inherited selection and cannot be modified directly. '
                    'Inherited selections are determined by the parent physics interface. '
                    'To change which domains this physics acts on, modify the physics-level selection '
                    f'via set_physics_selection with the physics interface tag "{physics_tag}" instead of a feature tag.'
                )
        except AttributeError:
            pass  # isInherited() not available in all COMSOL versions
        sel.set(entities)
    except (ValueError, LookupError):
        raise
    except Exception as exc:
        err_str = str(exc).lower()
        if "selection" in err_str or "edit" in err_str or "inherited" in err_str:
            raise ValueError(
                f'Cannot set selection on feature "{feature_tag}". '
                'This feature likely has an inherited or non-editable selection. '
                'Use list_physics_features to check selection_editable for each feature. '
                'To change the physics-level domain assignment, try setting selection on the physics interface itself.'
            ) from exc
        raise
    return {"tag": feature_tag, "entities": entities, "selection_set": True}


# ---------------------------------------------------------------------------
# Variable helpers
# ---------------------------------------------------------------------------
def _list_variables(model: Any, component: str = "") -> list[dict[str, Any]]:
    if component:
        var_container = model.java.component(component).variable()
    else:
        var_container = model.java.variable()
    tags = list(var_container.tags())
    result = []
    for tag in tags:
        var = var_container if not component else model.java.component(component).variable(tag)
        var_obj = var_container if not component else model.java.component(component).variable(tag)
        info: dict[str, Any] = {"tag": tag}
        try:
            info["name"] = str(var_obj.getString("name"))
        except Exception:
            pass
        try:
            info["expression"] = str(var_obj.getString("expr"))
        except Exception:
            pass
        result.append(info)
    return result


def _create_variable(
    model: Any, component: str, tag: str, name: str, expression: str,
) -> dict[str, Any]:
    if component:
        var_container = model.java.component(component).variable()
    else:
        var_container = model.java.variable()
    existing = list(var_container.tags())
    if tag in existing:
        return {"tag": tag, "created": False, "message": "Variable already exists."}
    var_container.create(tag)
    var_obj = var_container if not component else model.java.component(component).variable(tag)
    # For both global and component variables, set name and expression
    if component:
        var_node = model.java.component(component).variable(tag)
    else:
        var_node = model.java.variable(tag)
    var_node.set("name", name)
    var_node.set("expr", expression)
    return {"tag": tag, "name": name, "expression": expression, "created": True}


def _remove_variable(model: Any, component: str, tag: str) -> dict[str, Any]:
    if component:
        var_container = model.java.component(component).variable()
    else:
        var_container = model.java.variable()
    existing = list(var_container.tags())
    if tag not in existing:
        raise LookupError(f'Variable "{tag}" does not exist.')
    var_container.remove(tag)
    return {"tag": tag, "removed": True}
