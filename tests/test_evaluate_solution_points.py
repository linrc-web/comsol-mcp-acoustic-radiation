"""Offline fake-Java tests. No JVM, server connection or live model access."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import math
import pytest
from comsol_mcp import _point_ops as ops


class Array(list):
    pass


@pytest.fixture
def env(monkeypatch):
    arrays = []
    def conversion(kind):
        def make(value):
            arrays.append((kind, value))
            return Array(value)
        return make
    monkeypatch.setattr(ops, "_java_types", lambda: (
        conversion("String[]"), conversion("int[]"), conversion("double[][]"),
        int, conversion("int[][]")))
    selection = Mock(spec=["geom", "set"])
    feature = Mock(spec=["set", "setInterpolationCoordinates", "selection", "isComplex", "getData", "getImagData"])
    feature.selection.return_value = selection
    feature.isComplex.return_value = True
    feature.getData.return_value = [[[3., 0.]]]
    feature.getImagData.return_value = [[[4., -2.]]]
    tags = ["existing"]
    numerical = Mock(spec=["tags", "create", "remove"])
    numerical.tags.side_effect = lambda: list(tags)
    numerical.create.side_effect = lambda tag, kind: tags.append(tag)
    numerical.remove.side_effect = lambda tag: tags.remove(tag)
    ds = Mock(spec=["getType", "getString"])
    ds.getType.return_value = "Solution"
    ds.getString.side_effect = {"solution": "sol1", "comp": "comp1", "geom": "geom1"}.__getitem__
    datasets = Mock(spec=["tags"])
    datasets.tags.return_value = ["dset1"]
    result = Mock(spec=["dataset", "numerical"])
    result.dataset.side_effect = lambda *a: ds if a else datasets
    result.numerical.side_effect = lambda *a: feature if a else numerical
    info = Mock(spec=["isValid", "getOuterSolnum", "getSolnum", "getPNames", "getPvals", "getUnits", "getUnit"])
    info.isValid.return_value = True
    info.getOuterSolnum.return_value = []
    info.getSolnum.return_value = [1]
    info.getPNames.return_value = [["freq"]]
    info.getPvals.return_value = [[1e6]]
    info.getUnits.return_value = [["Hz"]]
    info.getUnit.return_value = "Hz"
    solution = Mock(spec=["isInitialized", "isEmpty", "getSolutioninfo", "getPNames", "getPVals"])
    solution.isInitialized.return_value = True
    solution.isEmpty.return_value = False
    solution.getSolutioninfo.return_value = info
    solution.getPNames.return_value = ["freq"]
    def parameter_values(solnum):
        if solnum != 1:
            raise ValueError("Invalid solution number")
        return [1e6]
    solution.getPVals.side_effect = parameter_values
    sols = Mock(spec=["tags"])
    sols.tags.return_value = ["sol1"]
    geom = Mock(spec=["getSDim", "getNDomains", "lengthUnit"])
    geom.getSDim.return_value = 3
    geom.getNDomains.return_value = 2
    geom.lengthUnit.return_value = "m"
    geoms = Mock(spec=["tags"])
    geoms.tags.return_value = ["geom1"]
    comp = Mock(spec=["geom"])
    comp.geom.side_effect = lambda *a: geom if a else geoms
    comps = Mock(spec=["tags"])
    comps.tags.return_value = ["comp1"]
    java = Mock(spec=["result", "sol", "component"])
    java.result.return_value = result
    java.sol.side_effect = lambda *a: solution if a else sols
    java.component.side_effect = lambda *a: comp if a else comps
    args = dict(expressions=["comp1.acpr.p_t"], coordinates=[[1, 2, 3], [4, 5, 6]],
                dataset="dset1", solution_tag="sol1", component="comp1", geometry="geom1",
                domain_ids=[1], solnum=1, expected_frequency_hz=1e6)
    return NS(**locals())


def run(e, **kwargs):
    return ops._evaluate_solution_points(NS(java=e.java), **(e.args | kwargs))


@pytest.mark.parametrize("changes", [
    {"expressions": []}, {"expressions": [""]}, {"expressions": [1]},
    {"coordinates": []}, {"coordinates": [[1, 2]]}, {"coordinates": [[1, 2, 3, 4]]},
    {"coordinates": [[True, 0, 0]]}, {"coordinates": [[float('nan'), 0, 0]]},
    {"coordinate_unit": "cm"}, {"solnum": 0}, {"solnum": True},
    {"expected_frequency_hz": 0}, {"expected_frequency_hz": float('inf')},
    {"expected_frequency_hz": 2e6}, {"solnum": 2}, {"outersolnum": 0},
    {"outersolnum": 2}, {"domain_ids": [3]}, {"domain_ids": [1, 1]},
    {"dataset": "missing"}, {"solution_tag": "missing"},
    {"component": "missing"}, {"geometry": "missing"},
])
def test_input_failure_before_creation(env, changes):
    with pytest.raises(ValueError):
        run(env, **changes)
    env.numerical.create.assert_not_called()


@pytest.mark.parametrize("unit,scale", [("m", 1), ("mm", 1e-3), ("um", 1e-6), ("nm", 1e-9)])
def test_units_arrays_configuration_and_complex(env, unit, scale):
    data = run(env, coordinate_unit=unit)
    env.feature.setInterpolationCoordinates.assert_called_once_with(
        Array([[scale, 4*scale], [2*scale, 5*scale], [3*scale, 6*scale]]))
    calls = env.feature.set.call_args_list
    assert (("data", "dset1"), {}) in calls
    assert (("expr", Array(env.args["expressions"])), {}) in calls
    assert (("solnum", Array([1])), {}) in calls
    env.selection.geom.assert_called_once_with("geom1", 3)
    env.selection.set.assert_called_once_with(Array([1]))
    assert {k for k, v in env.arrays} == {"String[]", "int[]", "double[][]"}
    value = data["results"][0]["values"][0]
    assert value == dict(real=3., imag=4., magnitude=5., phase_rad=math.atan2(4, 3))
    env.feature.getImagData.assert_called_once()
    tag, kind = env.numerical.create.call_args.args
    assert tag.startswith("mcp_interp_") and tag != "existing" and kind == "Interp"
    env.numerical.remove.assert_called_once_with(tag)
    assert env.tags == ["existing"] and data["structural_tree_restored"]
    assert not any(c.args[0] == "outersolnum" for c in calls)


def test_real(env):
    env.feature.isComplex.return_value = False
    assert run(env)["results"][0]["values"][0]["imag"] == 0
    env.feature.getImagData.assert_not_called()


def test_outer(env):
    env.info.getOuterSolnum.return_value = [1, 2]
    with pytest.raises(ValueError, match="explicit"):
        run(env)
    run(env, outersolnum=2)
    env.feature.set.assert_any_call("outersolnum", 2)
    assert env.info.getPvals.call_args.args[0] == [[2, 1]]


@pytest.mark.parametrize("value", [[], [[[1]]], [[[1, 2]], [[1, 2]]], [[[float('nan'), 2]]]])
def test_shape_and_nonfinite_cleanup(env, value):
    env.feature.getData.return_value = value
    with pytest.raises(RuntimeError):
        run(env)
    assert env.tags == ["existing"]


def test_evaluation_failure_cleanup(env):
    env.feature.getData.side_effect = RuntimeError("evaluation failed")
    with pytest.raises(RuntimeError, match="evaluation failed"):
        run(env)
    assert env.tags == ["existing"]


def test_cleanup_failure(env):
    env.numerical.remove.side_effect = RuntimeError("cannot remove")
    with pytest.raises(RuntimeError, match="removal failed.*not restored"):
        run(env)


def test_creation_failure_no_delete(env):
    env.numerical.create.side_effect = RuntimeError("create failed")
    with pytest.raises(RuntimeError, match="create failed"):
        run(env)
    env.numerical.remove.assert_not_called()


def test_collision(env, monkeypatch):
    env.tags.append("mcp_interp_collision")
    ids = iter([NS(hex="collision"), NS(hex="new")])
    monkeypatch.setattr(ops.uuid, "uuid4", lambda: next(ids))
    run(env)
    env.numerical.create.assert_called_once_with("mcp_interp_new", "Interp")
    assert env.tags == ["existing", "mcp_interp_collision"]


@pytest.mark.parametrize("target,method,value", [
    ("ds", "getType", "CutPoint3D"), ("geom", "getSDim", 2),
    ("geom", "lengthUnit", "mm"), ("solution", "isEmpty", True),
    ("info", "isValid", False), ("solution", "getPNames", ["time"]),
    ("info", "getUnit", "kHz"),
])
def test_metadata_fail_closed(env, target, method, value):
    getattr(getattr(env, target), method).return_value = value
    with pytest.raises(ValueError):
        run(env)
    env.numerical.create.assert_not_called()


def test_guard_and_registration(monkeypatch):
    from comsol_mcp import _tools_solution as tools
    from comsol_mcp._server import SAFE_VISIBLE_MAIN_WRITE_TOOLS, SAFE_READ_TOOLS
    assert "evaluate_solution_points" in SAFE_VISIBLE_MAIN_WRITE_TOOLS
    assert "evaluate_solution_points" not in SAFE_READ_TOOLS
    def guard(name):
        assert name == "evaluate_solution_points"
        raise RuntimeError("identity mismatch")
    monkeypatch.setattr(tools, "_require_visible_main", guard)
    monkeypatch.setattr(tools, "_run_tool", lambda name, callback: callback())
    with pytest.raises(RuntimeError, match="identity mismatch"):
        tools.evaluate_solution_points(["p"], [[0, 0, 0]], "d", "s", "c", "g", [1], 1, 1e6)


def test_multiple_expressions_preserve_order(env):
    env.feature.getData.return_value = [[[3., 0.]], [[10., 20.]]]
    env.feature.getImagData.return_value = [[[4., -2.]], [[0., 0.]]]
    data = run(env, expressions=["background", "total"])
    assert [r["expression"] for r in data["results"]] == ["background", "total"]
    assert [v["real"] for v in data["results"][1]["values"]] == [10., 20.]


@pytest.mark.parametrize("property,value", [("solution", "sol2"), ("comp", "comp2"), ("geom", "geom2")])
def test_dataset_mapping(env, property, value):
    props = {"solution": "sol1", "comp": "comp1", "geom": "geom1", property: value}
    env.ds.getString.side_effect = props.__getitem__
    with pytest.raises(ValueError, match="Dataset"):
        run(env)
    env.numerical.create.assert_not_called()


def test_imaginary_shape_failure(env):
    env.feature.getImagData.return_value = []
    with pytest.raises(RuntimeError, match="shape"):
        run(env)
    assert env.tags == ["existing"]


def test_config_failure_cleanup(env):
    env.feature.set.side_effect = RuntimeError("setter failed")
    with pytest.raises(RuntimeError, match="setter failed"):
        run(env)
    assert env.tags == ["existing"]


def test_cleanup_inventory_unavailable(env):
    env.numerical.tags.side_effect = [["existing"], RuntimeError("lost connection")]
    with pytest.raises(RuntimeError, match="Cannot verify cleanup"):
        run(env)
    env.numerical.remove.assert_called_once()


def test_explicit_jpype_factories(monkeypatch):
    import sys
    calls = []
    def factory(kind, dimensions=1):
        calls.append((kind, dimensions))
        return (kind, dimensions)
    fake = NS(JArray=factory, JString="string", JInt="int", JDouble="double")
    monkeypatch.setitem(sys.modules, "jpype", fake)
    assert ops._java_types() == (("string", 1), ("int", 1), ("double", 2), "int", ("int", 2))
    assert calls == [("string", 1), ("int", 1), ("double", 2), ("int", 2)]


class FakeComsolError(RuntimeError):
    def stacktrace(self):
        return "fake Java stack"


COMSOL_MESSAGE = "com.comsol.util.exceptions.FlException: Unknown solution.\nMessages:\n\tUnknown solution.\n\t- Tag\n"


@pytest.mark.parametrize("target,method,stage,owned", [
    ("java", "sol", "solution_lookup", False),
    ("result", "dataset", "dataset_lookup", False),
    ("ds", "getString", "dataset_solution_reference_read", False),
    ("info", "getOuterSolnum", "outer_numbers", False),
    ("numerical", "create", "interp_create", False),
    ("feature", "getData", "interp_get_data", True),
    ("feature", "getImagData", "interp_get_imag_data", True),
    ("numerical", "remove", "interp_remove", True),
    ("feature", "isComplex", "interp_is_complex", True),
    ("solution", "isEmpty", "solution_empty", False),
    ("ds", "getType", "dataset_type", False),
    ("solution", "getPNames", "solver_parameter_names", False),
    ("solution", "getPVals", "solver_parameter_values", False),
    ("info", "getUnit", "solver_parameter_units", False),
])
def test_exact_failure_diagnostics(env, target, method, stage, owned):
    getter = getattr(getattr(env, target), method)
    original = getter.side_effect
    def fail(*args):
        if (target, method) in [("java", "sol"), ("result", "dataset")] and not args:
            return original(*args)
        raise FakeComsolError(COMSOL_MESSAGE)
    getter.side_effect = fail
    with pytest.raises(RuntimeError) as caught:
        run(env)
    diagnostic = caught.value.point_diagnostics
    assert diagnostic["stage"] == stage + ".before"
    first = diagnostic["failures"][0]
    assert first["message"] == COMSOL_MESSAGE
    assert first["java_exception_string"] == COMSOL_MESSAGE
    assert first["exception_class"].endswith("FakeComsolError")
    assert "Traceback (most recent call last)" in first["python_traceback"]
    assert COMSOL_MESSAGE in str(caught.value)
    assert diagnostic["temporary_interp_owned"] is owned
    assert diagnostic["cleanup_attempted"] is owned
    assert diagnostic["cleanup_succeeded"] is (False if method == "remove" else True if owned else None)
    assert diagnostic["numerical_tags_before"] == (["existing"] if owned or method == "create" else None)
    if owned and method != "remove":
        assert diagnostic["numerical_tags_after"] == ["existing"]
    elif method == "remove":
        assert len(diagnostic["numerical_tags_after"]) == 2
    elif method == "create":
        assert diagnostic["numerical_tags_after"] == ["existing"]
    else:
        assert diagnostic["numerical_tags_after"] is None
        env.numerical.create.assert_not_called()
    if not owned:
        env.numerical.remove.assert_not_called()
    # The serialized diagnostic survives the existing MCP str(exception) envelope.
    import json
    assert json.loads(str(caught.value).split(" | point_diagnostics=", 1)[1]) == diagnostic


def test_frequency_failure_stage(env):
    with pytest.raises(ValueError) as caught:
        run(env, expected_frequency_hz=2e6)
    d = caught.value.point_diagnostics
    assert d["stage"] == "solver_frequency_validation.before"
    assert d["failures"][0]["exception_class"] == "builtins.ValueError"
    assert not d["temporary_interp_owned"] and not d["cleanup_attempted"]
    env.numerical.create.assert_not_called()


def test_primary_failure_survives_cleanup_failure(env):
    env.feature.getData.side_effect = FakeComsolError(COMSOL_MESSAGE)
    env.numerical.remove.side_effect = RuntimeError("remove failed")
    with pytest.raises(RuntimeError) as caught:
        run(env)
    d = caught.value.point_diagnostics
    assert d["stage"] == "interp_get_data.before"
    assert [f["stage"] for f in d["failures"]] == [
        "interp_get_data.before", "interp_remove.before", "numerical_inventory_verify.before"]
    assert d["cleanup_attempted"] and d["cleanup_succeeded"] is False


def test_cleanup_inventory_failure_stage(env):
    env.numerical.tags.side_effect = [["existing"], RuntimeError("inventory unavailable")]
    with pytest.raises(RuntimeError) as caught:
        run(env)
    d = caught.value.point_diagnostics
    assert d["stage"] == "numerical_inventory_after.before"
    assert d["cleanup_attempted"] and d["cleanup_succeeded"] is True
    assert d["numerical_tags_before"] == ["existing"] and d["numerical_tags_after"] is None


def test_success_diagnostics_do_not_change_payload_or_java_calls(env):
    d = ops._PointDiagnostics()
    data = ops._evaluate_solution_points(NS(java=env.java), **env.args, _diagnostic=d)
    assert set(data) == {"dataset", "solution_tag", "component", "geometry", "solnum",
        "outersolnum", "frequency_hz", "coordinates_m", "complex", "results",
        "numerical_tags_before", "numerical_tags_after", "structural_tree_restored", "state_note"}
    assert not d.failures
    assert d.cleanup_attempted and d.cleanup_succeeded
    assert d.before == d.after == ["existing"]
    for name in ["solution_tags", "dataset_tags", "solution_lookup", "dataset_lookup",
                 "dataset_solution_reference_read", "component_lookup", "geometry_lookup",
                 "geometry_dimension", "solution_initialized", "solution_empty", "solution_info",
                 "outer_numbers", "solver_parameter_names", "solver_parameter_values",
                 "solver_parameter_units", "solver_frequency_validation", "numerical_inventory_before", "interp_create",
                 "interp_dataset", "interp_expressions", "interp_coordinates",
                 "interp_geometry_selection", "interp_domain_selection", "interp_solnum",
                 "interp_optional_outersolnum", "interp_get_data", "interp_is_complex",
                 "interp_get_imag_data", "result_shape_real", "result_shape_imag",
                 "interp_remove", "numerical_inventory_after", "numerical_inventory_verify"]:
        assert name + ".before" in d.checkpoints and name + ".after" in d.checkpoints
    assert d.checkpoints.index("interp_is_complex.before") < d.checkpoints.index("interp_get_data.before")
    env.info.getSolnum.assert_not_called()
    env.info.getPNames.assert_not_called()
    env.info.getPvals.assert_not_called()
    env.info.getUnits.assert_not_called()
    env.solution.getPNames.assert_called_once_with()
    env.solution.getPVals.assert_called_once_with(1)
    env.info.getUnit.assert_called_once_with("freq")
    assert data["outersolnum"] == 1
    assert d.checkpoints.index("solver_frequency_validation.after") < d.checkpoints.index("interp_create.before")
    env.java.sol.assert_any_call("sol1")
    assert env.java.sol.call_count == 2
    env.feature.getData.assert_called_once_with()
    assert env.numerical.tags.call_count == 2
    # No solver, study or save API calls can be added unnoticed.
    assert {call[0] for call in env.java.mock_calls} <= {
        "result", "sol", "component", "result().dataset", "result().numerical"}


def test_identity_diagnostic_returned_through_mcp_envelope(monkeypatch):
    import json
    from comsol_mcp import _tools_solution as tools
    def guard(name):
        raise FakeComsolError(COMSOL_MESSAGE)
    def envelope(name, callback):
        try:
            callback()
        except Exception as exc:
            return {"success": False, "error": str(exc)}
    monkeypatch.setattr(tools, "_require_visible_main", guard)
    monkeypatch.setattr(tools, "_run_tool", envelope)
    result = tools.evaluate_solution_points(["p"], [[0, 0, 0]], "d", "s", "c", "g", [1], 1, 1e6)
    assert result["success"] is False
    d = json.loads(result["error"].split(" | point_diagnostics=", 1)[1])
    assert d["stage"] == "current_model_identity.before"
    assert d["numerical_tags_before"] is None and not d["cleanup_attempted"]


def test_worker_failure_stage_not_overwritten_by_cleanup(env, monkeypatch):
    from comsol_mcp import _dispatch
    job = NS(stage="lock")
    monkeypatch.setattr(_dispatch, "current_job", lambda: job)
    env.feature.getData.side_effect = FakeComsolError(COMSOL_MESSAGE)
    with pytest.raises(RuntimeError) as caught:
        run(env)
    assert job.stage == "interp_get_data.before"
    d = caught.value.point_diagnostics
    assert d["checkpoints"][-1] == "numerical_inventory_verify.after"
    assert d["cleanup_succeeded"] is True


@pytest.mark.parametrize("names,values,unit", [
    ([], [], "Hz"),
    (["time"], [1e6], "Hz"),
    (["freq", "freq"], [1e6, 1e6], "Hz"),
    (["freq"], [], "Hz"),
    (["freq"], [1e6, 2e6], "Hz"),
    (["freq"], [1e6], "kHz"),
    (["freq"], [float("nan")], "Hz"),
    (["freq"], [float("inf")], "Hz"),
    (["freq"], [float("-inf")], "Hz"),
])
def test_direct_frequency_metadata_rejected_before_interp(env, names, values, unit):
    env.solution.getPNames.return_value = names
    env.solution.getPVals.side_effect = lambda selected: values
    env.info.getUnit.return_value = unit
    with pytest.raises(ValueError) as caught:
        run(env)
    d = caught.value.point_diagnostics
    assert d["stage"] == "solver_frequency_validation.before"
    assert not d["temporary_interp_owned"] and not d["cleanup_attempted"]
    env.numerical.create.assert_not_called()
    env.info.getPNames.assert_not_called()
    env.info.getPvals.assert_not_called()
    env.info.getUnits.assert_not_called()


def test_second_inner_frequency_uses_typed_requested_index(env, monkeypatch):
    class JavaInt(int):
        pass
    factories = ops._java_types()
    monkeypatch.setattr(ops, "_java_types", lambda: (*factories[:3], JavaInt, factories[4]))
    def values(selected):
        assert type(selected) is JavaInt
        return {1: [1e6], 2: [2e6]}[selected]
    env.solution.getPVals.side_effect = values
    # Tuple metadata must not even be consulted for a no-outer solution.
    for method in (env.info.getSolnum, env.info.getPNames, env.info.getPvals, env.info.getUnits):
        method.side_effect = AssertionError("No outer tuple exists")
    data = run(env, solnum=2, expected_frequency_hz=2e6)
    assert data["solnum"] == 2 and data["frequency_hz"] == 2e6
    env.solution.getPVals.assert_called_once_with(2)
    env.feature.set.assert_any_call("solnum", Array([2]))
    env.numerical.create.assert_called_once()
    # Only stored metadata getters; no run/solve method is available or called.
    assert {call[0] for call in env.solution.mock_calls} == {
        "isInitialized", "isEmpty", "getSolutioninfo", "getPNames", "getPVals",
        "getSolutioninfo().isValid", "getSolutioninfo().getOuterSolnum", "getSolutioninfo().getUnit"}
    with pytest.raises(ValueError, match="frequency"):
        run(env, solnum=1, expected_frequency_hz=2e6)
    assert env.numerical.create.call_count == 1


@pytest.mark.parametrize("outers,selected,valid", [
    ([1, 2], None, False), ([1, 2], 3, False),
    ([2], None, True), ([1, 2], 2, True),
])
def test_outer_branch_keeps_tuple_metadata_and_checkpoints(env, outers, selected, valid):
    env.info.getOuterSolnum.return_value = outers
    d = ops._PointDiagnostics()
    if not valid:
        with pytest.raises(ValueError):
            ops._evaluate_solution_points(NS(java=env.java), **(env.args | {"outersolnum": selected}), _diagnostic=d)
        env.numerical.create.assert_not_called()
        env.info.getPNames.assert_not_called()
    else:
        data = ops._evaluate_solution_points(NS(java=env.java), **(env.args | {"outersolnum": selected}), _diagnostic=d)
        outer = selected if selected is not None else outers[0]
        assert data["outersolnum"] == outer
        env.info.getSolnum.assert_called_once_with(outer, True)
        env.info.getPNames.assert_called_once_with([[outer, 1]])
        env.info.getPvals.assert_called_once_with([[outer, 1]])
        env.info.getUnits.assert_called_once_with([[outer, 1]])
        for checkpoint in ("inner_solnum", "parameter_tuple", "parameter_names", "parameter_values", "parameter_units", "frequency_metadata", "frequency_match"):
            assert checkpoint + ".after" in d.checkpoints
        assert d.checkpoints.index("frequency_match.after") < d.checkpoints.index("interp_create.before")
    env.solution.getPNames.assert_not_called()
    env.solution.getPVals.assert_not_called()
    env.info.getUnit.assert_not_called()


def test_outer_missing_inner_solution_fails_before_creation(env):
    env.info.getOuterSolnum.return_value = [1]
    env.info.getSolnum.return_value = [1]
    with pytest.raises(ValueError, match="solnum does not exist"):
        run(env, solnum=2)
    env.numerical.create.assert_not_called()


def test_outer_original_parameter_exception_diagnostics(env):
    env.info.getOuterSolnum.return_value = [1]
    env.info.getPNames.side_effect = FakeComsolError(COMSOL_MESSAGE)
    with pytest.raises(RuntimeError) as caught:
        run(env)
    d = caught.value.point_diagnostics
    assert d["stage"] == "parameter_names.before"
    assert d["failures"][0]["java_exception_string"] == COMSOL_MESSAGE
    assert "FakeComsolError" in d["failures"][0]["python_traceback"]
    env.numerical.create.assert_not_called()
