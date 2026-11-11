"""Guarded temporary interpolation; no solver or dataset mutations."""
from __future__ import annotations

import json
import math
import uuid
import traceback
from contextlib import contextmanager



class _PointDiagnostics:
    """Per-call diagnostics; no Java queries or additional timeout checks."""
    def __init__(self):
        self.stage = "input_validation.before"
        self.checkpoints = []
        self.failures = []
        self._last_exception = None
        self.owned = False
        self.cleanup_attempted = False
        self.cleanup_succeeded = None
        self.before = None
        self.after = None

    def mark(self, stage):
        self.checkpoints.append(stage)
        self._publish_stage(stage)

    def _publish_stage(self, stage):
        self.stage = stage
        from comsol_mcp._dispatch import current_job
        job = current_job()
        if job is not None:
            job.stage = stage

    def capture(self, exc):
        if exc is self._last_exception:
            return
        self._last_exception = exc
        self.failures.append(dict(stage=self.stage,
            exception_class=type(exc).__module__ + "." + type(exc).__qualname__,
            message=str(exc), python_traceback=traceback.format_exc(),
            java_exception_string=str(exc) if hasattr(exc, "__javaclass__") or hasattr(exc, "stacktrace") else None,
            temporary_interp_owned=self.owned))

    @contextmanager
    def step(self, name):
        self.mark(name + ".before")
        try:
            yield
        except Exception as exc:
            if exc is not self._last_exception:
                self.stage = name + ".before"
            self.capture(exc)
            raise
        else:
            self.mark(name + ".after")

    def call(self, name, callback):
        with self.step(name):
            return callback()

    def run(self, callback):
        try:
            return callback()
        except Exception as exc:
            if hasattr(exc, "point_diagnostics"):
                raise
            if not self.failures:
                self.capture(exc)
            diagnostic = dict(stage=self.failures[0]["stage"], failures=self.failures,
                checkpoints=self.checkpoints, temporary_interp_owned=self.owned,
                cleanup_attempted=self.cleanup_attempted, cleanup_succeeded=self.cleanup_succeeded,
                numerical_tags_before=self.before, numerical_tags_after=self.after)
            # Cleanup has its own checkpoints; the outer MCP failure envelope
            # must still identify the original failure rather than cleanup's end.
            self._publish_stage(diagnostic["stage"])
            # MCP envelopes preserve str(exc), including this structured JSON.
            kind = ValueError if isinstance(exc, ValueError) else RuntimeError
            error = kind(str(exc) + " | point_diagnostics=" + json.dumps(diagnostic))
            error.point_diagnostics = diagnostic
            raise error from exc


def _java_types():
    from jpype import JArray, JString, JInt, JDouble
    return JArray(JString), JArray(JInt), JArray(JDouble, 2), JInt, JArray(JInt, 2)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Coordinates and frequency must be finite numbers.")
    return float(value)


def _evaluate_solution_points(model, expressions, coordinates, dataset, solution_tag,
                              component, geometry, domain_ids, solnum,
                              expected_frequency_hz, coordinate_unit="m", outersolnum=None,
                              *, _diagnostic=None):
    diagnostic = _diagnostic if _diagnostic is not None else _PointDiagnostics()
    return diagnostic.run(lambda: _evaluate_solution_points_impl(
        model, expressions, coordinates, dataset, solution_tag, component, geometry,
        domain_ids, solnum, expected_frequency_hz, coordinate_unit, outersolnum,
        diagnostic=diagnostic))


def _evaluate_solution_points_impl(model, expressions, coordinates, dataset, solution_tag,
                              component, geometry, domain_ids, solnum,
                              expected_frequency_hz, coordinate_unit="m", outersolnum=None, *, diagnostic):
    with diagnostic.step("input_validation"):
        if not isinstance(expressions, list) or not expressions or any(
            not isinstance(e, str) or not e.strip() for e in expressions
        ):
            raise ValueError("expressions must be a non-empty list of non-empty strings.")
        scales = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
        if coordinate_unit not in scales:
            raise ValueError("coordinate_unit must be m, mm, um, or nm.")
        if not isinstance(coordinates, list) or not coordinates:
            raise ValueError("coordinates must be a non-empty Nx3 list.")
        points = []
        for point in coordinates:
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError("coordinates must be a non-empty Nx3 list.")
            points.append([_number(v) * scales[coordinate_unit] for v in point])
        if any(not math.isfinite(v) for point in points for v in point):
            raise ValueError("Converted coordinates are not finite.")
        if type(solnum) is not int or solnum < 1:
            raise ValueError("solnum must be a positive integer.")
        with diagnostic.step("input_outersolnum"):
            if outersolnum is not None and (type(outersolnum) is not int or outersolnum < 1):
                raise ValueError("outersolnum must be a positive integer or null.")
        frequency = _number(expected_frequency_hz)
        if frequency <= 0:
            raise ValueError("expected_frequency_hz must be positive.")
        if not isinstance(domain_ids, list) or not domain_ids or any(
            type(d) is not int or d < 1 for d in domain_ids
        ) or len(set(domain_ids)) != len(domain_ids):
            raise ValueError("domain_ids must contain unique positive integers.")
        for name, tag in (("dataset", dataset), ("solution_tag", solution_tag),
                          ("component", component), ("geometry", geometry)):
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError(f"{name} is required.")

    with diagnostic.step("model_java"):
        java = model.java
    with diagnostic.step("dataset_tags"):
        if dataset not in list(java.result().dataset().tags()):
            raise ValueError("Dataset does not exist.")
    with diagnostic.step("solution_tags"):
        if solution_tag not in list(java.sol().tags()):
            raise ValueError("Solution does not exist.")
    with diagnostic.step("dataset_lookup"):
        ds = java.result().dataset(dataset)
    with diagnostic.step("dataset_solution_reference"):
        if (str(diagnostic.call("dataset_type", ds.getType)) != "Solution"
                or str(diagnostic.call("dataset_solution_reference_read",
                                       lambda: ds.getString("solution"))) != solution_tag):
            raise ValueError("Dataset must be a Solution dataset linked to solution_tag.")
    with diagnostic.step("dataset_component"):
        if str(ds.getString("comp")) != component:
            raise ValueError("Dataset component does not match component.")
    with diagnostic.step("component_tags"):
        if component not in list(java.component().tags()):
            raise ValueError("Component does not exist.")
    with diagnostic.step("component_lookup"):
        comp = java.component(component)
    with diagnostic.step("geometry_tags"):
        if geometry not in list(comp.geom().tags()):
            raise ValueError("Geometry does not exist.")
    with diagnostic.step("geometry_lookup"):
        geom = comp.geom(geometry)
    with diagnostic.step("geometry_dimension"):
        if int(geom.getSDim()) != 3:
            raise ValueError("Geometry must be 3D.")
    with diagnostic.step("dataset_geometry"):
        if str(ds.getString("geom")) != geometry:
            raise ValueError("Dataset geometry does not match geometry.")
    with diagnostic.step("domain_validation"):
        if any(d > int(geom.getNDomains()) for d in domain_ids):
            raise ValueError("domain_ids contain an invalid domain.")
    # Interp coordinates have no unit property: fail closed unless the existing
    # geometry uses meters. Never change its length unit to accommodate input.
    with diagnostic.step("geometry_length_unit"):
        if str(geom.lengthUnit()) != "m":
            raise ValueError("Only existing geometries with lengthUnit=m are supported.")
    with diagnostic.step("solution_lookup"):
        solution = java.sol(solution_tag)
    with diagnostic.step("solution_initialized_empty"):
        if (not diagnostic.call("solution_initialized", solution.isInitialized)
                or diagnostic.call("solution_empty", solution.isEmpty)):
            raise ValueError("Solution is not initialized or is empty.")
    with diagnostic.step("solution_info"):
        info = solution.getSolutioninfo()
    with diagnostic.step("solution_info_valid"):
        if not info.isValid():
            raise ValueError("Solution metadata is invalid.")
    with diagnostic.step("java_types"):
        strings, ints, doubles2, jint, ints2 = _java_types()
    with diagnostic.step("outer_numbers"):
        outer_values = [int(v) for v in info.getOuterSolnum()]
    with diagnostic.step("outer_validation"):
        if outersolnum is None and len(outer_values) > 1:
            raise ValueError("Multiple outer solutions require explicit outersolnum.")
    outer = outersolnum if outersolnum is not None else (outer_values[0] if outer_values else None)
    with diagnostic.step("outer_selection_validation"):
        if (outer_values and outer not in outer_values
                or not outer_values and outersolnum not in (None, 1)):
            raise ValueError("Invalid outersolnum.")
    if not outer_values:
        # No outer tuple exists. The indexed solver getter selects the requested
        # one-based inner solution and rejects unavailable solution numbers.
        with diagnostic.step("solver_parameter_names"):
            names = [str(v) for v in solution.getPNames()]
        with diagnostic.step("solver_parameter_values"):
            values = [float(v) for v in solution.getPVals(jint(solnum))]
        with diagnostic.step("solver_parameter_units"):
            units = [str(info.getUnit(name)) for name in names]
        with diagnostic.step("solver_frequency_validation"):
            if names.count("freq") != 1 or len(values) != len(names) or len(units) != len(names):
                raise ValueError("Cannot identify an unambiguous stored frequency.")
            index = names.index("freq")
            if (units[index] != "Hz" or not math.isfinite(values[index])
                    or not math.isclose(values[index], frequency, rel_tol=1e-9, abs_tol=1e-6)):
                raise ValueError("Stored frequency does not match expected_frequency_hz in Hz.")
    else:
        with diagnostic.step("inner_solnum"):
            if solnum not in [int(v) for v in info.getSolnum(jint(outer), True)]:
                raise ValueError("solnum does not exist in the selected outer solution.")
        with diagnostic.step("parameter_tuple"):
            pair = ints2([[outer, solnum]])
        with diagnostic.step("parameter_names"):
            names = [str(v) for v in info.getPNames(pair)[0]]
        with diagnostic.step("parameter_values"):
            values = [float(v) for v in info.getPvals(pair)[0]]
        with diagnostic.step("parameter_units"):
            units = [str(v) for v in info.getUnits(pair)[0]]
        with diagnostic.step("frequency_metadata"):
            if names.count("freq") != 1 or len(values) != len(names) or len(units) != len(names):
                raise ValueError("Cannot identify an unambiguous stored frequency.")
        index = names.index("freq")
        with diagnostic.step("frequency_match"):
            if units[index] != "Hz" or not math.isclose(values[index], frequency, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("Stored frequency does not match expected_frequency_hz in Hz.")
    with diagnostic.step("numerical_lookup"):
        numerical = java.result().numerical()
    with diagnostic.step("numerical_inventory_before"):
        before = [str(t) for t in numerical.tags()]
        diagnostic.before = before
    tag = "mcp_interp_" + uuid.uuid4().hex
    while tag in before:
        tag = "mcp_interp_" + uuid.uuid4().hex
    owned = False
    error = None
    data = None
    cleanup_error = None
    after = None
    try:
        with diagnostic.step("interp_create"):
            numerical.create(tag, "Interp")
        owned = True
        diagnostic.owned = True
        with diagnostic.step("interp_lookup"):
            feature = java.result().numerical(tag)
        with diagnostic.step("interp_dataset"):
            feature.set("data", dataset)
        with diagnostic.step("interp_expressions"):
            feature.set("expr", strings(expressions))
        with diagnostic.step("interp_coordinates"):
            feature.setInterpolationCoordinates(doubles2([list(axis) for axis in zip(*points)]))
        with diagnostic.step("interp_geometry_selection"):
            feature.selection().geom(geometry, jint(3))
        with diagnostic.step("interp_domain_selection"):
            feature.selection().set(ints(domain_ids))
        with diagnostic.step("interp_solnum"):
            feature.set("solnum", ints([solnum]))
        with diagnostic.step("interp_optional_outersolnum"):
            if outersolnum is not None or outer_values:
                feature.set("outersolnum", jint(outer))
        with diagnostic.step("interp_phase"):
            feature.set("phase", 0.0)
        with diagnostic.step("interp_extension"):
            feature.set("ext", 0.0)
        with diagnostic.step("interp_coordinate_errors"):
            feature.set("coorderr", "on")
        with diagnostic.step("interp_math_errors"):
            feature.set("matherr", "on")
        with diagnostic.step("interp_is_complex"):
            complex_result = bool(feature.isComplex())
        with diagnostic.step("interp_get_data"):
            real = feature.getData()
        with diagnostic.step("interp_get_imag_data"):
            imag = feature.getImagData() if complex_result else None

        def copy(array):
            if len(array) != len(expressions) or any(
                len(row) != 1 or len(row[0]) != len(points) for row in array
            ):
                raise ValueError("Unexpected Interp shape; expected [expressions][1][points].")
            result = [[float(v) for v in row[0]] for row in array]
            if any(not math.isfinite(v) for row in result for v in row):
                raise ValueError("Interp returned non-finite values (possibly outside selected domains).")
            return result

        with diagnostic.step("result_shape_real"):
            real = copy(real)
        with diagnostic.step("result_shape_imag"):
            imag = copy(imag) if complex_result else [[0.0] * len(points) for _ in expressions]
        rows = []
        for e, expression in enumerate(expressions):
            rows.append({"expression": expression, "values": [
                {"real": r, "imag": i, "magnitude": math.hypot(r, i), "phase_rad": math.atan2(i, r)}
                for r, i in zip(real[e], imag[e])]})
        data = dict(dataset=dataset, solution_tag=solution_tag, component=component,
                    geometry=geometry, solnum=solnum,
                    # Preserve the existing response convention for no outer sweep.
                    outersolnum=outer if outer is not None else 1,
                    frequency_hz=values[index], coordinates_m=points,
                    complex=complex_result, results=rows)
        with diagnostic.step("result_serialization"):
            json.dumps(data, allow_nan=False)  # Prepare serialization inside cleanup scope.
    except Exception as exc:
        error = str(exc)
    finally:
        if owned:
            try:
                with diagnostic.step("interp_remove"):
                    diagnostic.cleanup_attempted = True
                    numerical.remove(tag)
                    diagnostic.cleanup_succeeded = True
            except Exception as exc:
                diagnostic.cleanup_succeeded = False
                cleanup_error = "Temporary Interp removal failed: " + str(exc)
        try:
            with diagnostic.step("numerical_inventory_after"):
                after = [str(t) for t in numerical.tags()]
                diagnostic.after = after
            with diagnostic.step("numerical_inventory_verify"):
                if after != before:
                    cleanup_error = (cleanup_error or "") + " Numerical inventory was not restored."
                    diagnostic.capture(RuntimeError("Numerical inventory was not restored."))
        except Exception as exc:
            cleanup_error = (cleanup_error or "") + " Cannot verify cleanup: " + str(exc)
    if error or cleanup_error:
        raise RuntimeError("; ".join(v for v in (error, cleanup_error) if v))
    data.update(numerical_tags_before=before, numerical_tags_after=after,
                structural_tree_restored=True,
                state_note="Temporary numerical node removed; byte-identical internal state is not guaranteed.")
    return data
