#!/usr/bin/env python3
"""Pure model operation helpers: parameters, expressions, metrics, tree, geometry."""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------
def _parameter_rows(model: Any) -> list[dict[str, str]]:
    names = list(model.java.param().varnames())
    rows = []
    for name in names:
        key = str(name)
        rows.append(
            {
                "name": key,
                "expression": str(model.java.param().get(key)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Expression evaluation helpers
# ---------------------------------------------------------------------------
def _coerce_eval_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_eval_value(item) for item in value]
    return str(value)


def _last_scalar(value: Any) -> Any:
    current = value
    while isinstance(current, list) and current:
        current = current[-1]
    return current


def _evaluate_named_expressions(model: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each expression entry must be an object.")
        name = str(item.get("name", "")).strip()
        expression = str(item.get("expression", "")).strip()
        if not name:
            raise ValueError("Expression name is required.")
        if not expression:
            raise ValueError(f'Expression is required for "{name}".')
        row: dict[str, Any] = {"name": name, "expression": expression}
        try:
            raw = model.evaluate(expression)
            value = _coerce_eval_value(raw)
            row["value"] = value
            row["last_value"] = _coerce_eval_value(_last_scalar(value))
            row["ok"] = True
        except Exception as exc:
            row["ok"] = False
            row["error"] = str(exc)
        results.append(row)
    return results


def _clear_numerical(model: Any) -> None:
    numerical = model.java.result().numerical()
    for tag in list(numerical.tags()):
        numerical.remove(str(tag))


def _numeric_result(name: str, expression: str, value: Any = None, *, ok: bool = True, error: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {"name": name, "expression": expression, "ok": ok}
    if ok:
        row["value"] = value
        row["last_value"] = _coerce_eval_value(_last_scalar(value))
    else:
        row["error"] = error or "NA"
    return row


# ---------------------------------------------------------------------------
# Core metrics helpers
# ---------------------------------------------------------------------------
def _find_initialized_solution_tag(model: Any) -> str:
    for tag in list(model.java.sol().tags()):
        tag_str = str(tag)
        try:
            if model.java.sol(tag_str).isInitialized():
                return tag_str
        except Exception:
            continue
    return ""


def _read_last_time_day(model: Any, sol_tag: str) -> float | None:
    if not sol_tag:
        return None
    try:
        values = model.java.sol(sol_tag).getPVals()
        if values is not None and len(values) > 0:
            last_value = float(values[-1])
            return last_value / 86400.0 if abs(last_value) > 1e3 else last_value
    except Exception:
        return None
    return None


def _eval_global_last(model: Any, expression: str) -> float | None:
    try:
        _clear_numerical(model)
        numerical_list = model.java.result().numerical()
        numerical_list.create("gev1", "EvalGlobal")
        feature = model.java.result().numerical("gev1")
        feature.set("expr", [expression])
        feature.setIndex("looplevelinput", "last", 0)
        feature.run()
        return float(feature.getReal()[0][0])
    except Exception:
        return None


def _eval_domain_average_last(model: Any, expression: str, domains: list[int]) -> float | None:
    try:
        _clear_numerical(model)
        numerical_list = model.java.result().numerical()
        numerical_list.create("dom1", "IntSurface")
        feature = model.java.result().numerical("dom1")
        feature.selection().geom("geom1", 2)
        feature.selection().set(domains)
        feature.set("intvolume", True)
        feature.set("expr", [expression, "1"])
        feature.setIndex("looplevelinput", "last", 0)
        feature.run()
        values = feature.getReal()
        return float(values[0][0] / values[1][0])
    except Exception:
        return None


def _eval_boundary_average_last(model: Any, expression: str, boundaries: list[int]) -> float | None:
    try:
        _clear_numerical(model)
        numerical_list = model.java.result().numerical()
        numerical_list.create("int1", "IntLine")
        feature = model.java.result().numerical("int1")
        feature.selection().geom("geom1", 1)
        feature.selection().set(boundaries)
        feature.set("intsurface", True)
        feature.set("expr", [expression, "1"])
        feature.setIndex("looplevelinput", "last", 0)
        feature.run()
        values = feature.getReal()
        return float(values[0][0] / values[1][0])
    except Exception:
        return None


def _eval_extremum_last(model: Any, feature_type: str, expression: str, domains: list[int]) -> float | None:
    try:
        _clear_numerical(model)
        numerical_list = model.java.result().numerical()
        numerical_list.create("ext1", feature_type)
        feature = model.java.result().numerical("ext1")
        feature.selection().geom("geom1", 2)
        feature.selection().set(domains)
        feature.set("expr", [expression])
        feature.setIndex("looplevelinput", "last", 0)
        feature.run()
        return float(feature.getReal()[0][0])
    except Exception:
        return None


def _get_model_dimension(model: Any) -> int:
    """Get the configured or auto-detected spatial dimension of the model.

    Priority:
    1. Workflow state `model_dimension` (set by configure_single_main_workflow)
    2. Auto-detect from existing physics interfaces
    3. Default to 0 (unknown)
    """
    # Check workflow state first
    try:
        from comsol_mcp._state import _read_workflow_state
        state = _read_workflow_state()
        dim = int(state.get("model_dimension", 0))
        if dim in (1, 2, 3):
            return dim
    except Exception:
        pass

    # Auto-detect from existing physics interfaces
    try:
        comps = list(model.java.component().tags())
        if comps:
            comp = model.java.component(comps[0])
            phys_tags = list(comp.physics().tags())
            if phys_tags:
                phys = comp.physics(phys_tags[0])
                # Try common API methods for getting spatial dimension from physics
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
                # Try reading the physics selection dimension
                try:
                    sel = phys.feature().tags()
                    if sel:
                        feat = phys.feature(sel[0])
                        # The selection geom dimension reveals spatial dim
                        for attr_name in ("_geomDim", "dim"):
                            try:
                                d = int(getattr(feat.selection(), attr_name))
                                if d in (1, 2, 3):
                                    return d
                            except Exception:
                                continue
                except Exception:
                    pass
    except Exception:
        pass

    return 0


def _evaluate_aggregate(
    model: Any,
    expression: str,
    aggregate: str,
    domains: list[int] | None = None,
    boundaries: list[int] | None = None,
    time_point: str = "last",
) -> Any:
    """Evaluate an expression with aggregation. Returns scalar for aggregated, raw for 'none'.

    Automatically selects correct COMSOL result feature type based on model dimension:
    - 3D: domains→Volume, boundaries→Surface
    - 2D: domains→Surface, boundaries→Line
    - 1D: domains→Line, boundaries→Point
    """
    aggregate = (aggregate or "none").lower().strip()
    if aggregate == "none":
        raw = model.evaluate(expression)
        return _coerce_eval_value(raw)

    _clear_numerical(model)
    numerical_list = model.java.result().numerical()

    # Get spatial dimension from workflow config or auto-detect
    dim = _get_model_dimension(model)
    if dim == 0:
        dim = 2  # Default to 2D if unknown (common for structural mechanics)

    # Map: (dim, entity) → suffix for max/min/int
    # 3D: domains=Volume, boundaries=Surface
    # 2D: domains=Surface, boundaries=Line
    # 1D: domains=Line, boundaries=Point (rare)
    if boundaries:
        geom_dim = max(dim - 1, 1)  # boundary is (dim-1) entity
    elif domains:
        geom_dim = dim
    else:
        geom_dim = dim

    dim_suffix = {3: "Volume", 2: "Surface", 1: "Line"}.get(geom_dim, "Surface")

    # Choose correct COMSOL result feature type
    if aggregate == "max":
        if domains or boundaries:
            tag, ftype = "agg1", f"Max{dim_suffix}"
        else:
            tag, ftype = "agg1", f"Max{dim_suffix}"
    elif aggregate == "min":
        if domains or boundaries:
            tag, ftype = "agg1", f"Min{dim_suffix}"
        else:
            tag, ftype = "agg1", f"Min{dim_suffix}"
    elif aggregate == "avg":
        if domains or boundaries:
            tag, ftype = "agg1", f"Int{dim_suffix}"
        else:
            tag, ftype = "agg1", "EvalGlobal"
    elif aggregate == "integral":
        tag, ftype = "agg1", f"Int{dim_suffix}"
    else:
        raise ValueError(f'Unknown aggregate "{aggregate}". Use max, min, avg, integral, or none.')

    numerical_list.create(tag, ftype)
    feature = model.java.result().numerical(tag)

    # Set selection with correct geometric dimension
    if domains:
        feature.selection().geom("geom1", dim)
        feature.selection().set(domains)
    elif boundaries:
        feature.selection().geom("geom1", max(dim - 1, 1))
        feature.selection().set(boundaries)

    # Set expression(s) based on aggregate type
    if aggregate == "avg" and (domains or boundaries):
        feature.set("expr", [expression, "1"])
    else:
        feature.set("expr", [expression])

    if time_point and time_point != "all":
        feature.setIndex("looplevelinput", time_point, 0)

    feature.run()
    values = feature.getReal()

    if aggregate == "avg" and (domains or boundaries):
        denom = values[1][0]
        if denom == 0:
            return float("nan")
        return float(values[0][0] / denom)
    return float(values[0][0])


# ---------------------------------------------------------------------------
# Model tree
# ---------------------------------------------------------------------------
def _model_tree_data(model: Any) -> dict[str, Any]:
    from comsol_mcp._state import _safe_model_label, _safe_model_path
    java = model.java
    components = [str(tag) for tag in java.component().tags()]
    component_details = []
    for component in components:
        comp = java.component(component)
        component_details.append(
            {
                "tag": component,
                "geometries": [str(tag) for tag in comp.geom().tags()],
                "meshes": [str(tag) for tag in comp.mesh().tags()],
                "physics": [str(tag) for tag in comp.physics().tags()],
                "materials": [str(tag) for tag in comp.material().tags()],
            }
        )
    return {
        "label": _safe_model_label(model),
        "file_path": _safe_model_path(model),
        "components": components,
        "component_details": component_details,
        "parameters": [row["name"] for row in _parameter_rows(model)],
        "studies": [str(tag) for tag in java.study().tags()],
        "solutions": [str(tag) for tag in java.sol().tags()],
        "datasets": [str(tag) for tag in java.result().dataset().tags()],
        "results": [str(tag) for tag in java.result().tags()],
    }


# ---------------------------------------------------------------------------
# Geometry feature helpers
# ---------------------------------------------------------------------------
def _normalize_properties(properties_json: str) -> list[tuple[str, str | list[str]]]:
    """Preserve scalar ``value`` versus array ``values``, including length one."""
    raw = str(properties_json or "").strip()
    if not raw:
        return []
    parsed = json.loads(raw)
    items: list[dict[str, Any]]
    if isinstance(parsed, dict):
        items = []
        for name, value in parsed.items():
            if isinstance(value, list):
                items.append({"name": name, "values": [str(item) for item in value]})
            else:
                items.append({"name": name, "value": str(value)})
    elif isinstance(parsed, list):
        items = parsed
    else:
        raise ValueError("properties_json must be a JSON object or array.")

    normalized: list[tuple[str, str | list[str]]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each properties_json entry must be an object.")
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        if "values" in item:
            values = item.get("values")
            if not isinstance(values, list):
                raise ValueError(f'Property "{name}" values must be an array.')
            normalized.append((name, [str(value) for value in values]))
        elif "value" in item:
            normalized.append((name, str(item.get("value", ""))))
        else:
            raise ValueError(f'Property "{name}" must include value or values.')
    return normalized


def _ensure_component_java(model: Any, component: str, dimension: int) -> dict[str, Any]:
    java = model.java
    if component not in list(java.component().tags()):
        java.component().create(component, True)
    return {"component": component, "dimension": dimension}


def _ensure_geometry_java(model: Any, component: str, geometry: str, dimension: int) -> dict[str, Any]:
    java = model.java
    if component not in list(java.component().tags()):
        java.component().create(component, True)
    if dimension <= 0:
        dimension = 2
    if geometry not in list(java.component(component).geom().tags()):
        java.component(component).geom().create(geometry, dimension)
    return {"component": component, "geometry": geometry, "dimension": dimension}


def _ensure_mesh_java(model: Any, component: str, mesh: str) -> dict[str, Any]:
    java = model.java
    if component not in list(java.component().tags()):
        java.component().create(component, True)
    if mesh not in list(java.component(component).mesh().tags()):
        java.component(component).mesh().create(mesh)
    return {"component": component, "mesh": mesh}


def _geometry_boundary_info(
    model: Any, component: str, geometry: str, boundary_ids: list[int],
) -> dict[str, Any]:
    """Read final geometry only; never build, select, or retry.

    COMSOL 6.4 GeomInfo.getVertexCoord() is sdim-by-number-of-vertices.
    Entity IDs are 1-based; coordinate columns use zero-based array indexing.
    https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/api/com/comsol/model/GeomInfo.html
    """
    for name, value in (("component", component), ("geometry", geometry)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required.")
    if not isinstance(boundary_ids, list) or not boundary_ids:
        raise ValueError("boundary_ids must be a non-empty list of integers.")
    if any(type(value) is not int for value in boundary_ids):
        raise ValueError("boundary_ids must contain integers only (not booleans).")
    if any(value <= 0 for value in boundary_ids):
        raise ValueError("boundary_ids must be positive 1-based entity IDs.")
    if len(set(boundary_ids)) != len(boundary_ids):
        raise ValueError("boundary_ids must not contain duplicate IDs.")

    g = model.java.component(component).geom(geometry)
    boundary_count = int(g.getNBoundaries())
    for boundary_id in boundary_ids:
        if boundary_id > boundary_count:
            raise ValueError(
                f"boundary_id {boundary_id} exceeds boundary_count {boundary_count}."
            )
    coordinates = [[float(value) for value in row] for row in g.getVertexCoord()]
    space_dimension = len(coordinates)
    if space_dimension != 3:
        raise ValueError("Final geometry must be 3D for boundary-to-domain adjacency (2, 3).")
    vertex_count = len(coordinates[0])
    if any(len(row) != vertex_count for row in coordinates):
        raise ValueError("getVertexCoord() returned a nonrectangular coordinate matrix.")
    bounding_box = [float(value) for value in g.getBoundingBox()]
    boundaries = []
    for boundary_id in boundary_ids:
        adjacent_domains = [int(value) for value in g.getAdjExt(2, 3, boundary_id)]
        vertex_ids = [int(value) for value in g.getAdj(2, 0, boundary_id)]
        if any(value <= 0 or value > vertex_count for value in vertex_ids):
            raise ValueError(
                f"getAdj(2, 0, {boundary_id}) returned a vertex ID outside 1..{vertex_count}."
            )
        boundaries.append({
            "boundary_id": boundary_id,
            "adjacent_domains_ext": adjacent_domains,
            "vertex_ids": vertex_ids,
            "vertex_coordinates": [
                [coordinates[axis][vertex_id - 1] for axis in range(space_dimension)]
                for vertex_id in vertex_ids
            ],
        })
    return {
        "component": component, "geometry": geometry,
        "space_dimension": space_dimension,
        "geometry_bounding_box": bounding_box, "boundaries": boundaries,
    }


def _geometry_topology_info(
    model: Any, component: str, geometry: str, object_name: str | None = None,
) -> dict[str, Any]:
    """Read built topology, without selection access, evaluation or builds.

    COMSOL 6.4 GeomInfo is inherited by both GeomSequence and GeomObject:
    https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/api/com/comsol/model/GeomInfo.html
    Object entity numbering is local, not finalized geometry numbering.
    Missing getter bindings are unavailable; Java exceptions propagate once.
    """
    for name, value in (("component", component), ("geometry", geometry)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required.")
    if object_name is not None and (
        not isinstance(object_name, str) or not object_name.strip()
    ):
        raise ValueError("object_name must be a non-empty string or null.")
    geom = model.java.component(component).geom(geometry)
    info = geom
    if object_name is not None:
        info = geom.obj(object_name)
        if info is None or not info.exists():
            raise LookupError(
                f'Geometry object "{object_name}" does not exist in the current build state.'
            )
    result = dict(component=component, geometry=geometry,
                  scope="final" if object_name is None else "object",
                  object_name=object_name, unavailable={})

    def convert(value, scalar, depth):
        if value is None:
            return None
        if depth:
            return [convert(item, scalar, depth - 1) for item in value]
        return scalar(value)

    # Fixed GeomInfo getter allowlist, valid for both scopes. No arbitrary API calls.
    getters = (
        ("space_dimension", "getSDim", int, 0),
        ("domain_count", "getNDomains", int, 0),
        ("boundary_count", "getNBoundaries", int, 0),
        ("entity_counts", "getNEntities", int, 1),
        ("bounding_box", "getBoundingBox", float, 1),
        ("finite_void_count", "getNFiniteVoids", int, 0),
        ("voids_are_labeled", "voidsAreLabeled", bool, 0),
        ("up_down_ext", "getUpDownExt", int, 2),
    )
    for field, method, scalar, depth in getters:
        getter = getattr(info, method, None)
        if getter is None:
            result[field] = None
            result["unavailable"][field] = (
                f"{method} is unavailable on the {result['scope']} GeomInfo binding."
            )
            continue
        value = getter()  # No retry or fallback after an API exception.
        result[field] = convert(value, scalar, depth)
        if value is None:
            result["unavailable"][field] = f"{method} returned null."
    return result


def _geometry_feature_object_names(
    model: Any, component: str, geometry: str, feature_tag: str,
) -> dict[str, Any]:
    """Read current output names with objectNames(); never build or retry."""
    for name, value in (("component", component), ("geometry", geometry),
                        ("feature_tag", feature_tag)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required.")
    geom = model.java.component(component).geom(geometry)
    if feature_tag not in list(geom.feature().tags()):
        raise LookupError(f'Geometry feature "{feature_tag}" does not exist.')
    names = geom.feature(feature_tag).objectNames()
    if names is None:
        raise RuntimeError("Geometry feature objectNames() returned null.")
    return {"component": component, "geometry": geometry,
            "feature_tag": feature_tag, "object_names": [str(name) for name in names]}


def _list_geometry_feature_properties(
    model: Any, component: str, geometry: str, feature_tag: str,
) -> dict[str, Any]:
    """Enumerate existing properties via COMSOL 6.4 PropFeature.properties().

    API: https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/api/com/comsol/model/PropFeature.html#properties()
    Only navigation, tags(), and properties() are called; no fallback or retry.
    """
    for name, value in (("component", component), ("geometry", geometry),
                        ("feature_tag", feature_tag)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required.")
    geom = model.java.component(component).geom(geometry)
    if feature_tag not in list(geom.feature().tags()):
        raise LookupError(f'Geometry feature "{feature_tag}" does not exist.')
    names = geom.feature(feature_tag).properties()
    if names is None:
        raise RuntimeError("Geometry feature properties() returned null.")
    return {"component": component, "geometry": geometry,
            "feature_tag": feature_tag, "property_names": [str(name) for name in names]}


def _read_geometry_feature_properties(
    model: Any, component: str, geometry: str, feature_tag: str,
    property_names: list[str],
    include_string_representation: bool = False,
) -> dict[str, Any]:
    """Read existing properties with COMSOL 6.4 PropFeature typed getters.

    API: https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/api/com/comsol/model/PropFeature.html
    No ensure/create, setters, selection access, evaluation, or build calls.
    DoubleRowMatrix uses the documented string-array representation to retain
    unevaluated expressions when COMSOL stores them. An optional string-array
    probe does not guarantee recovery of original expressions for other types.
    """
    for name, value in (("component", component), ("geometry", geometry),
                        ("feature_tag", feature_tag)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required.")
    if (not isinstance(property_names, list) or not property_names
            or any(not isinstance(name, str) or not name.strip()
                   for name in property_names)):
        raise ValueError("property_names must be a non-empty array of non-empty strings.")
    geom = model.java.component(component).geom(geometry)
    if feature_tag not in list(geom.feature().tags()):
        raise LookupError(f'Geometry feature "{feature_tag}" does not exist.')
    feature = geom.feature(feature_tag)
    # Explicit allowlist: never derive an arbitrary Java method from user input.
    getters = {
        "String": ("getString", str, 0),
        "StringArray": ("getStringArray", str, 1),
        "StringMatrix": ("getStringMatrix", str, 2),
        "Boolean": ("getBoolean", bool, 0),
        "BooleanArray": ("getBooleanArray", bool, 1),
        "BooleanMatrix": ("getBooleanMatrix", bool, 2),
        "Int": ("getInt", int, 0),
        "IntArray": ("getIntArray", int, 1),
        "IntMatrix": ("getIntMatrix", int, 2),
        "Double": ("getDouble", float, 0),
        "DoubleArray": ("getDoubleArray", float, 1),
        "DoubleMatrix": ("getDoubleMatrix", float, 2),
        "DoubleRowMatrix": ("getStringArray", str, 1),
    }

    def convert(value, scalar, depth):
        if value is None:
            return None  # Preserve Java null rather than inventing a value.
        if depth:
            return [convert(item, scalar, depth - 1) for item in value]
        return scalar(value)

    properties = []
    for name in property_names:
        if not feature.hasProperty(name):
            raise LookupError(f'Geometry feature "{feature_tag}" has no property "{name}".')
        value_type = str(feature.getValueType(name))
        if value_type not in getters:
            raise ValueError(f'Unsupported COMSOL property type "{value_type}" for property "{name}".')
        getter, scalar, depth = getters[value_type]
        value = getattr(feature, getter)(name)
        item = {"name": name, "value_type": value_type,
                "getter": getter, "value": convert(value, scalar, depth)}
        if include_string_representation:
            item.update(string_getter=None, string_value=None,
                        string_available=False, string_error=None)
            if depth == 1:
                item["string_getter"] = "getStringArray"
                try:
                    string_value = feature.getStringArray(name)
                    item["string_value"] = convert(string_value, str, 1)
                    item["string_available"] = string_value is not None
                    if string_value is None:
                        item["string_error"] = "getStringArray returned null."
                except Exception as exc:
                    item["string_error"] = str(exc)
            else:
                item["string_error"] = "String-array probe is only available for vector properties."
        properties.append(item)
    return {"component": component, "geometry": geometry,
            "feature_tag": feature_tag, "properties": properties}


def _apply_feature_properties(feature: Any, properties: list[tuple[str, str | list[str]]]) -> list[dict[str, Any]]:
    applied = []
    for name, values in properties:
        if isinstance(values, str):
            feature.set(name, values)
            applied.append({"name": name, "value": values})
        else:
            feature.set(name, values)
            applied.append({"name": name, "values": values})
    return applied


def _read_solution_dof_field(model, sol_tag, field_name, solnum, offset, limit):
    """Join documented DOF metadata to raw solution values; never evaluate nodes."""
    import math

    availability = {}
    data = dict(sol_tag=sol_tag, field_name=field_name, solnum=solnum,
                available_field_names=[], getters=availability, status="unavailable",
                error="", total_matching_dofs=None, page_count=0,
                offset=offset, limit=limit, points=[],
                coordinate_units=dict(coords="model length unit", g_coords="geometry length unit"),
                value_scope="raw DOF coefficients; no interpolation or expression evaluation")

    def read(name, getter, convert=lambda v: v):
        try:
            value = getter()
            if value is None:
                raise ValueError("Java getter returned null")
            value = convert(value)
            availability[name] = {"status": "available"}
            return value
        except Exception as exc:
            availability[name] = {"status": "unavailable", "error": str(exc)}
            return None

    def finite(value):
        value = float(value)
        return value if math.isfinite(value) else str(value)

    sol = read("solution", lambda: model.java.sol(sol_tag))
    if sol is None:
        return data
    xmi = read("xmesh_info", lambda: sol.xmeshInfo())
    if xmi is None:
        return data
    dofs = read("dofs", lambda: xmi.dofs())
    if dofs is None:
        return data
    names = read("dof_names", lambda: dofs.dofNames(), lambda v: [str(x) for x in v])
    if names is None:
        return data
    data["available_field_names"] = names
    if field_name not in names:
        data.update(status="field_not_found", error="field_name must exactly match an available DOF name.",
                    total_matching_dofs=0)
        return data

    if solnum is None:
        solnum = read("default_solnum", lambda: sol.getDefaultSolnum(), int)
    data["solnum"] = solnum
    if solnum is None:
        return data
    if solnum < 1:
        data.update(status="invalid", error="Default solution number must be positive.")
        return data

    u = read("u_real", lambda: sol.getU(solnum), list)
    ui = read("u_imag", lambda: sol.getUImag(solnum), list)
    ni = read("name_inds", lambda: dofs.nameInds(), list)
    si = read("solution_vector_inds", lambda: dofs.solVectorInds(), list)
    coords = read("coords", lambda: dofs.coords(), lambda v: [list(row) for row in v])
    gc = read("g_coords", lambda: dofs.gCoords(), lambda v: [list(row) for row in v])
    geoms = read("geom_nums", lambda: dofs.geomNums(), list)
    nodes = read("nodes", lambda: dofs.nodes(), list)

    try:
        # Validate every available array, including DOFs outside the requested page/field.
        arrays = [("name_inds", ni), ("solution_vector_inds", si),
                  ("geom_nums", geoms), ("nodes", nodes)]
        lengths = [(name, len(values)) for name, values in arrays if values is not None]
        for name, matrix in (("coords", coords), ("g_coords", gc)):
            if matrix is not None:
                if not 1 <= len(matrix) <= 3:
                    raise ValueError(f"{name} must have 1..3 coordinate rows.")
                lengths.extend((name, len(row)) for row in matrix)
        if len({n for _, n in lengths}) > 1:
            raise ValueError("DOF array length mismatch: " + repr(lengths))
        if coords is not None and gc is not None and len(coords) != len(gc):
            raise ValueError("Coordinate dimension mismatch.")
        if u is not None and ui is not None and len(u) != len(ui):
            raise ValueError("Real/imaginary solution vector length mismatch.")
        if ni is not None:
            for index in ni:
                if int(index) != index or not 0 <= index < len(names):
                    raise ValueError("nameInds index out of range or noninteger.")
        if si is not None:
            for index in si:
                if int(index) != index or index < 0:
                    raise ValueError("solVectorInds index out of range or noninteger.")
                for vector in (u, ui):
                    if vector is not None and index >= len(vector):
                        raise ValueError("solVectorInds index out of range.")
        for name, values, minimum in (("geomNums", geoms, 1), ("nodes", nodes, 0)):
            if values is not None and any(int(v) != v or v < minimum for v in values):
                raise ValueError(f"Invalid {name} index.")
        matched = None if ni is None else [k for k, index in enumerate(ni) if names[int(index)] == field_name]
        if matched is not None:
            data["total_matching_dofs"] = len(matched)
        # Never emit an unvalidated coordinate/value association.
        if any(v is None for v in (ni, si, coords)) or (u is None and ui is None):
            return data
        for k in matched[offset:offset + limit]:
            j = int(si[k])
            point = dict(dof_index=k, solution_vector_index=j,
                         geom_num=None if geoms is None else int(geoms[k]),
                         node=None if nodes is None else int(nodes[k]),
                         p_real=None if u is None else finite(u[j]),
                         p_imag=None if ui is None else finite(ui[j]))
            point.update({axis: finite(coords[d][k]) if d < len(coords) else None
                          for d, axis in enumerate(("x", "y", "z"))})
            point["g_coords"] = None if gc is None else [finite(row[k]) for row in gc]
            data["points"].append(point)
        data["page_count"] = len(data["points"])
        data["status"] = "partial" if any(v["status"] == "unavailable" for v in availability.values()) else "available"
    except (ValueError, TypeError, OverflowError, IndexError) as exc:
        data.update(status="invalid", error=str(exc), points=[], page_count=0)
    return data


# COMSOL 6.4 documented getters only. No evaluation features are constructed.
# https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/api/com/comsol/model/SolverSequence.html
# https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_solver.51.10.html
def _inspect_solution(model: Any, sol_tag: str) -> dict[str, Any]:
    import math

    def number(value):
        value = float(value)
        # Keep the response valid JSON even for nonfinite COMSOL data.
        return value if math.isfinite(value) else str(value)

    def read(getter, convert):
        try:
            value = getter()
            if value is None:
                raise ValueError("Java getter returned null")
            return {"status": "available", **convert(value)}
        except Exception as exc:
            return {"status": "unavailable", "error": str(exc)}

    def vector(values):
        length = len(values)
        all_finite = True
        sample = []
        for index, item in enumerate(values):
            value = float(item)
            if not math.isfinite(value):
                all_finite = False
            if index < 8:
                sample.append(number(value))
        return {"length": length, "sample": sample, "all_finite": all_finite}

    # Access an existing sequence by tag; this overload never creates one.
    solution = model.java.sol(sol_tag)
    data = {
        "sol_tag": sol_tag,
        "vector_scope": "default solution number; raw solver degrees of freedom",
        "is_initialized": read(lambda: solution.isInitialized(), lambda v: {"value": bool(v)}),
        "pvals": read(lambda: solution.getPVals(),
                      lambda v: {"values": [number(x) for x in v], "length": len(v)}),
        "u_real": read(lambda: solution.getU(), vector),
        "u_imag": read(lambda: solution.getUImag(), vector),
    }
    try:
        info = solution.getSolutioninfo()
        if info is None:
            raise ValueError("getSolutioninfo returned null")
    except Exception as exc:
        data["solution_info"] = {"status": "unavailable", "error": str(exc)}
    else:
        data["solution_info"] = {
            "status": "available",
            "is_valid": read(lambda: info.isValid(), lambda v: {"value": bool(v)}),
            "levels": read(lambda: info.getLevels(), lambda v: {"value": int(v)}),
            "level_names": read(lambda: info.getLevelNames(), lambda v: {"values": [str(x) for x in v]}),
            "outer_solnum": read(lambda: info.getOuterSolnum(), lambda v: {"values": [int(x) for x in v]}),
        }
    return data


# Mesh lookup never creates sequences or features (including for getters).
def _mesh_sequence(model: Any, component: str, mesh: str) -> Any:
    for name, value in (("component", component), ("mesh", mesh)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required.")
    if component not in [str(t) for t in model.java.component().tags()]:
        raise LookupError(f'Component "{component}" does not exist.')
    comp = model.java.component(component)
    if mesh not in [str(t) for t in comp.mesh().tags()]:
        raise LookupError(f'Mesh "{mesh}" does not exist.')
    return comp.mesh(mesh)


def _mesh_feature(model: Any, component: str, mesh: str, feature_tag: str) -> Any:
    if not isinstance(feature_tag, str) or not feature_tag.strip():
        raise ValueError("feature_tag is required.")
    seq = _mesh_sequence(model, component, mesh)
    if feature_tag not in [str(t) for t in seq.feature().tags()]:
        raise LookupError(f'Mesh feature "{feature_tag}" does not exist.')
    return seq.feature(feature_tag)


def _read_mesh_feature_property_values(feature: Any, property_names: list[str]) -> list[dict[str, Any]]:
    """Compatibility wrapper for the shared read-only PropFeature reader."""
    return _read_feature_property_values(feature, property_names)


def _read_feature_property_values(feature: Any, property_names: list[str]) -> list[dict[str, Any]]:
    """Read PropFeature values using only COMSOL 6.4 documented typed getters.

    Verified against COMSOL64/plugins/com.comsol.api_1.0.0.jar and:
    https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/api/com/comsol/model/PropFeature.html
    getValueType selects one getter, called once; no probes or fallback.
    DoubleRowMatrix uses getStringArray to retain returned unevaluated strings.
    Numeric types return API numeric values, not reconstructed expressions.
    """
    if not isinstance(property_names, list) or any(
        not isinstance(name, str) or not name.strip() for name in property_names
    ):
        raise ValueError("property_names must be a list of non-empty strings.")
    getters = {
        "String": ("getString", str, 0, "string"),
        "StringArray": ("getStringArray", str, 1, "string[]"),
        "StringMatrix": ("getStringMatrix", str, 2, "string[][]"),
        "Boolean": ("getBoolean", bool, 0, "boolean"),
        "Int": ("getInt", int, 0, "integer"),
        "IntArray": ("getIntArray", int, 1, "integer[]"),
        "Double": ("getDouble", float, 0, "number"),
        "DoubleArray": ("getDoubleArray", float, 1, "number[]"),
        "DoubleMatrix": ("getDoubleMatrix", float, 2, "number[][]"),
        "DoubleRowMatrix": ("getStringArray", str, 1, "string[]"),
    }

    def convert(value, scalar, depth):
        if value is None:
            return None
        if depth:
            return [convert(item, scalar, depth - 1) for item in value]
        return scalar(value)

    result = []
    for name in property_names:
        item = dict(name=name, value=None, value_type=None, representation=None,
                    getter=None, available=False, error=None)
        try:
            if not feature.hasProperty(name):
                raise LookupError(f'Property "{name}" does not exist.')
            kind = feature.getValueType(name)
            if kind is None:
                raise RuntimeError("getValueType returned null.")
            item["value_type"] = kind = str(kind)
            if kind not in getters:
                raise ValueError(f'Unsupported COMSOL property type "{kind}".')
            method, scalar, depth, representation = getters[kind]
            item.update(getter=method, representation=representation)
            getter = getattr(feature, method, None)
            if not callable(getter):
                raise RuntimeError(f"{method} getter unavailable.")
            value = getter(name)
            if value is None:
                raise RuntimeError(f"{method} returned null.")
            item.update(value=convert(value, scalar, depth), available=True)
        except Exception as exc:
            item["error"] = str(exc)
        result.append(item)
    return result
