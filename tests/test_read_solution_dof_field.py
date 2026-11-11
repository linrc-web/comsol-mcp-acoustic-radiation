"""No COMSOL runtime: audit every Java call against getter-only fakes."""
import json
from types import SimpleNamespace

import pytest

import comsol_mcp._model as model_module
import comsol_mcp._state as state
import comsol_mcp._tools_solution as tools
from comsol_mcp._server import SAFE_READ_TOOLS
from tests.test_inspect_solution import ReadOnlyJava


@pytest.fixture
def runtime(monkeypatch):
    dofs = ReadOnlyJava(dofNames=["comp1.p", "comp1.u"], nameInds=[0, 1, 0, 0],
                       solVectorInds=[3, 0, 2, 1], coords=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                       gCoords=[[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]],
                       geomNums=[1, 1, 1, 1], nodes=[0, 1, 2, 3])
    xmi = ReadOnlyJava(dofs=dofs)
    sol = ReadOnlyJava(getDefaultSolnum=2, getU=[10, 20, 30, 40],
                       getUImag=[-10, -20, -30, -40], xmeshInfo=xmi)
    java = ReadOnlyJava(sol=sol)
    monkeypatch.setattr(model_module, "_require_model", lambda: SimpleNamespace(java=java))
    monkeypatch.setattr(model_module, "_read_workflow_state", lambda: {})
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error), allow_nan=False))
    yield SimpleNamespace(dofs=dofs, sol=sol, java=java, xmi=xmi)
    for obj in (java, sol, xmi, dofs):
        # Exact allowlist: solVectorInds is a getter despite its 'solve' prefix.
        assert all(name in obj.getters for name, _ in obj.calls)
        assert len(obj.calls) == len({name for name, _ in obj.calls})


def call(**kwargs):
    return json.loads(tools.read_solution_dof_field("sol1", kwargs.pop("field_name", "comp1.p"), **kwargs))


def test_mapping_and_complex_values(runtime):
    data = call()["data"]
    assert data["status"] == "available"
    assert data["available_field_names"] == ["comp1.p", "comp1.u"]
    assert data["solnum"] == 2
    assert data["total_matching_dofs"] == data["page_count"] == 3
    assert [p["dof_index"] for p in data["points"]] == [0, 2, 3]
    assert [p["solution_vector_index"] for p in data["points"]] == [3, 2, 1]
    assert [p["p_real"] for p in data["points"]] == [40, 30, 20]
    assert [p["p_imag"] for p in data["points"]] == [-40, -30, -20]
    assert data["points"][0] == dict(dof_index=0, solution_vector_index=3, geom_num=1,
                                    node=0, p_real=40, p_imag=-40, x=1, y=5, z=9,
                                    g_coords=[10, 50, 90])
    assert ("getU", (2,)) in runtime.sol.calls
    assert ("getUImag", (2,)) in runtime.sol.calls
    assert "read_solution_dof_field" in SAFE_READ_TOOLS


@pytest.mark.parametrize("name", ["p", "comp1.P", "comp1.p ", "missing"])
def test_exact_name_only(runtime, name):
    data = call(field_name=name)["data"]
    assert data["status"] == "field_not_found"
    assert data["available_field_names"] == ["comp1.p", "comp1.u"]
    assert data["points"] == []
    assert runtime.sol.calls == [("xmeshInfo", ())]


@pytest.mark.parametrize("offset,limit,indices", [(0, 1, [0]), (1, 2, [2, 3]), (3, 2, []), (99, 1, [])])
def test_pagination(runtime, offset, limit, indices):
    data = call(solnum=1, offset=offset, limit=limit)["data"]
    assert data["total_matching_dofs"] == 3
    assert data["page_count"] == len(indices)
    assert [p["dof_index"] for p in data["points"]] == indices
    assert not any(name == "getDefaultSolnum" for name, _ in runtime.sol.calls)


@pytest.mark.parametrize("getter,value", [("nameInds", [0, 2, 0, 0]),
    ("solVectorInds", [3, 4, 2, 1]), ("solVectorInds", [3, -1, 2, 1]),
    ("nameInds", [0, -1, 0, 0]), ("nodes", [0, -1, 2, 3]), ("geomNums", [0, 1, 1, 1])])
def test_invalid_indices_even_outside_field_and_page(runtime, getter, value):
    runtime.dofs.getters[getter] = value
    data = call(limit=1)["data"]
    assert data["status"] == "invalid" and data["error"]
    assert data["points"] == []


@pytest.mark.parametrize("getter,value", [("nameInds", [0]), ("solVectorInds", []),
    ("coords", [[1], [2], [3]]), ("gCoords", [[1, 2, 3, 4], []]),
    ("geomNums", [1]), ("nodes", [0])])
def test_array_lengths(runtime, getter, value):
    runtime.dofs.getters[getter] = value
    data = call()["data"]
    assert data["status"] == "invalid" and "length mismatch" in data["error"]
    assert data["points"] == []


def test_vector_lengths(runtime):
    runtime.sol.getters["getUImag"] = [0]
    assert call()["data"]["status"] == "invalid"


@pytest.mark.parametrize("getter,key", [("getUImag", "u_imag"), ("getU", "u_real"),
    ("gCoords", "g_coords"), ("coords", "coords"), ("nameInds", "name_inds"),
    ("solVectorInds", "solution_vector_inds"), ("geomNums", "geom_nums"), ("nodes", "nodes"),
    ("getDefaultSolnum", "default_solnum"), ("xmeshInfo", "xmesh_info"),
    ("dofs", "dofs"), ("dofNames", "dof_names")])
def test_unavailable_no_retries(runtime, getter, key):
    obj = next(o for o in (runtime.sol, runtime.xmi, runtime.dofs) if getter in o.getters)
    obj.getters[getter] = AttributeError("not available")
    data = call()["data"]
    assert data["getters"][key] == dict(status="unavailable", error="not available")
    if getter == "getUImag":
        assert data["status"] == "partial"
        assert all(p["p_imag"] is None for p in data["points"])


@pytest.mark.parametrize("kwargs", [dict(offset=-1), dict(limit=0), dict(limit=10001),
                                   dict(solnum=0), dict(field_name="")])
def test_validation_before_java(runtime, kwargs):
    assert not call(**kwargs)["success"]
    assert runtime.java.calls == []


def test_visible_main_guard(runtime, monkeypatch):
    monkeypatch.setattr(tools, "_require_visible_main", lambda _: (_ for _ in ()).throw(RuntimeError("lock mismatch")))
    assert not call()["success"]
    assert runtime.java.calls == []
