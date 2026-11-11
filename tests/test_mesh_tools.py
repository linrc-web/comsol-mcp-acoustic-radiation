"""Mesh tool contract tests. Fake Java only; never start COMSOL/JVM."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import comsol_mcp._tools_mesh as tools
import comsol_mcp._model as model_module
import comsol_mcp._state as state
from comsol_mcp._server import SAFE_READ_TOOLS, SAFE_VISIBLE_MAIN_WRITE_TOOLS


@pytest.fixture
def runtime(monkeypatch):
    selection = Mock(spec=["geom", "set", "dim", "entities"])
    selection.dim.return_value = 2
    selection.entities.return_value = [6, 7, 8, 9, 10, 11]
    feature = Mock(spec=["selection", "set", "getType", "label", "properties"])
    feature.selection.return_value = selection
    feature.getType.return_value = "Size"
    feature.label.return_value = "Size 1"
    feature.properties.return_value = ["custom", "hmax"]
    features = {"size_cuboid": feature}
    seq = Mock(spec=["feature", "create", "automatic"])
    seq.feature.side_effect = lambda tag=None: SimpleNamespace(tags=lambda: list(features)) if tag is None else features[tag]
    seq.create.side_effect = lambda tag, kind: features.__setitem__(tag, feature)
    comp = SimpleNamespace(mesh=lambda tag=None: SimpleNamespace(tags=lambda: ["mesh1"]) if tag is None else seq)
    model = SimpleNamespace(java=SimpleNamespace(component=lambda tag=None:
        SimpleNamespace(tags=lambda: ["comp1"]) if tag is None else comp))
    monkeypatch.setattr(model_module, "_require_model", lambda: model)
    monkeypatch.setattr(model_module, "_read_workflow_state", lambda: {})
    monkeypatch.setattr(model_module, "_append_operation", lambda *a: None)
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error)))
    return SimpleNamespace(**locals())


def invoke(name, *args):
    return json.loads(getattr(tools, name)(*args))


@pytest.mark.parametrize("kind,tag", [("FreeTet", "ftet1"), ("Size", "size_local")])
def test_create(runtime, kind, tag):
    r = invoke("create_mesh_feature", "comp1", "mesh1", tag, kind)
    assert r["success"] and r["data"]["created"]
    runtime.seq.create.assert_called_once_with(tag, kind)
    assert not invoke("create_mesh_feature", "comp1", "mesh1", tag, kind)["success"]
    assert runtime.seq.create.call_count == 1


def test_update_strings(runtime):
    props = {"custom": "on", "hmax": "40[um]", "hmin": "0.5[um]", "hgrad": "1.3"}
    props.update({k + "active": "on" for k in ["hmax", "hmin", "hgrad", "hcurve", "hnarrow"]})
    r = invoke("update_mesh_feature", "comp1", "mesh1", "size_cuboid", json.dumps(props))
    assert r["success"]
    assert runtime.feature.set.call_args_list == [call(k, v) for k, v in props.items()]
    assert invoke("update_mesh_feature", "comp1", "mesh1", "size_cuboid", '{"hmax":"2[um]"}')["success"]
    runtime.feature.set.assert_called_with("hmax", "2[um]")


@pytest.mark.parametrize("dim,ids", [(2, [6,7,8,9,10,11]), (3, [1]), (2, [])])
def test_selection(runtime, dim, ids):
    assert invoke("set_mesh_feature_selection", "comp1", "mesh1", "size_cuboid", dim, ids)["success"]
    assert runtime.selection.mock_calls == [call.geom(dim), call.set(ids)]


def test_readonly_getters(runtime):
    r = invoke("get_mesh_feature_properties", "comp1", "mesh1", "size_cuboid")
    assert r["success"]
    assert r["data"]["feature_type"] == "Size"
    assert r["data"]["label"] == "Size 1"
    assert r["data"]["property_names"] == ["custom", "hmax"]
    assert runtime.feature.mock_calls == [call.getType(), call.label(), call.properties()]
    runtime.feature.reset_mock()
    r = invoke("get_mesh_feature_selection", "comp1", "mesh1", "size_cuboid")
    assert r["data"]["entity_ids"] == [6,7,8,9,10,11]
    assert r["data"]["geometry_dimension"] == 2
    assert runtime.feature.mock_calls == [call.selection(), call.selection().dim(), call.selection().entities()]
    runtime.seq.create.assert_not_called()
    runtime.seq.automatic.assert_not_called()
    # Specs contain no all/run/build, so unintended mutation also fails the call.


@pytest.mark.parametrize("ids", [[], None])
def test_empty_vs_unavailable(runtime, ids):
    runtime.selection.entities.return_value = ids
    r = invoke("get_mesh_feature_selection", "comp1", "mesh1", "size_cuboid")
    assert r["data"]["entity_ids"] == ids
    assert ("entity_ids" in r["data"]["unavailable"]) == (ids is None)


def test_missing_getter(runtime):
    del runtime.selection.entities
    r = invoke("get_mesh_feature_selection", "comp1", "mesh1", "size_cuboid")
    assert r["success"] and "entity_ids" in r["data"]["unavailable"]


def test_automatic(runtime):
    assert invoke("set_mesh_automatic", "comp1", "mesh1", False)["success"]
    runtime.seq.automatic.assert_called_once_with(False)
    assert not invoke("set_mesh_automatic", "comp1", "mesh1", "false")["success"]
    assert runtime.seq.automatic.call_count == 1


@pytest.mark.parametrize("comp,mesh,tag", [("absent", "mesh1", "size_cuboid"),
    ("comp1", "absent", "size_cuboid"), ("comp1", "mesh1", "absent"), ("", "mesh1", "size_cuboid")])
def test_missing_nodes(runtime, comp, mesh, tag):
    for name, extra in [("get_mesh_feature_properties", ()), ("get_mesh_feature_selection", ()),
                        ("update_mesh_feature", ('{"hmax":"2[um]"}',)),
                        ("set_mesh_feature_selection", (2, [1]))]:
        assert not invoke(name, comp, mesh, tag, *extra)["success"]
    runtime.seq.create.assert_not_called()
    runtime.feature.set.assert_not_called()
    runtime.selection.set.assert_not_called()


@pytest.mark.parametrize("dim,ids", [(True, [1]), (4, [1]), (2, [True]), (2, [0]), (2, [1,1]), (2, "[1]")])
def test_invalid_selection(runtime, dim, ids):
    assert not invoke("set_mesh_feature_selection", "comp1", "mesh1", "size_cuboid", dim, ids)["success"]
    runtime.feature.selection.assert_not_called()


def test_bad_property_validation(runtime):
    for props in ['42', '[{"name":"hmax"}]', '[{"name":"hmax","values":"bad"}]']:
        assert not invoke("update_mesh_feature", "comp1", "mesh1", "size_cuboid", props)["success"]
    runtime.feature.set.assert_not_called()


@pytest.mark.parametrize("name,extra,target", [
    ("create_mesh_feature", ("Size",), "create"),
    ("update_mesh_feature", ('{"hmax":"2[um]"}',), "set"),
    ("set_mesh_feature_selection", (2,[1]), "selection_set"),
    ("set_mesh_automatic", (False,), "automatic"),
    ("get_mesh_feature_properties", (), "getType"),
    ("get_mesh_feature_selection", (), "dim"),
])
def test_java_error_once(runtime, name, extra, target):
    obj = {"create": runtime.seq.create, "set": runtime.feature.set,
           "selection_set": runtime.selection.set, "automatic": runtime.seq.automatic,
           "getType": runtime.feature.getType, "dim": runtime.selection.dim}[target]
    obj.side_effect = RuntimeError("original Java error")
    args = ("comp1", "mesh1") + (() if name=="set_mesh_automatic" else
            ("newtag" if name=="create_mesh_feature" else "size_cuboid",))
    r = invoke(name, *args, *extra)
    assert not r["success"] and r["error"] == "original Java error"
    assert obj.call_count == 1


@pytest.mark.parametrize("name,extra", [
    ("create_mesh_feature", ("FreeTet",)), ("update_mesh_feature", ('{"hmax":"2[um]"}',)),
    ("set_mesh_feature_selection", (2,[1])), ("set_mesh_automatic", (False,))])
def test_guard_mismatch(runtime, monkeypatch, name, extra):
    monkeypatch.setattr(model_module, "_read_workflow_state", lambda: {"visible_main_locked": True, "guard_level": "strict"})
    monkeypatch.setattr(model_module, "_visible_main_mismatch", lambda *a: ["tag mismatch"])
    monkeypatch.setattr(model_module, "_visible_main_identity", lambda *a: {})
    args = ("comp1", "mesh1") + (() if name=="set_mesh_automatic" else ("size_cuboid",))
    r = invoke(name, *args, *extra)
    assert not r["success"] and "lock mismatch" in r["error"]
    assert runtime.seq.mock_calls == []
    assert runtime.feature.mock_calls == []


def test_registration_and_classification():
    registry = Mock()
    tools.register(registry)
    names = {c.args[0].__name__ for c in registry.add_tool.call_args_list}
    assert names == {"list_mesh_features", "create_mesh_feature", "update_mesh_feature", "set_mesh_feature_selection",
                     "get_mesh_feature_properties", "get_mesh_feature_property_values",
                     "get_mesh_feature_selection", "set_mesh_automatic",
                     "get_mesh_feature_active", "set_mesh_feature_active", "move_mesh_feature"}
    for name in names:
        if name.startswith(("get_", "list_")):
            assert name in SAFE_READ_TOOLS and name not in SAFE_VISIBLE_MAIN_WRITE_TOOLS
        else:
            assert name in SAFE_VISIBLE_MAIN_WRITE_TOOLS and name not in SAFE_READ_TOOLS


def test_list_mesh_features_order_and_readonly(runtime):
    runtime.features.clear()
    expected = []
    for index, (tag, kind, label) in enumerate([
        ("z_last_alphabetically", "SizeExpression", "Size Expression 1"),
        ("a_first_alphabetically", "FreeTet", "Actual label"),
    ]):
        feature = Mock(spec=["getType", "label"])
        feature.getType.return_value = kind
        feature.label.return_value = label
        runtime.features[tag] = feature
        expected.append(dict(index=index, tag=tag, type=kind, label=label))
    result = invoke("list_mesh_features", "comp1", "mesh1")
    assert result["success"]
    assert result["data"] == dict(component="comp1", mesh="mesh1", features=expected)
    assert runtime.seq.mock_calls == [call.feature(), *[call.feature(r["tag"]) for r in expected]]
    for feature in runtime.features.values():
        assert feature.mock_calls == [call.getType(), call.label()]
    runtime.selection.assert_not_called()
    runtime.selection.set.assert_not_called()
    # Exact call lists and restrictive specs reject run/runAll/create/remove/move/set,
    # automatic/selection changes, and every other unexpected model operation.


@pytest.mark.parametrize("component,mesh", [("comp1", "missing"), ("missing", "mesh1"), ("", "mesh1")])
def test_list_mesh_features_missing_sequence(runtime, component, mesh):
    result = invoke("list_mesh_features", component, mesh)
    assert not result["success"] and result["error"]
    assert runtime.seq.mock_calls == []


def test_list_mesh_features_empty(runtime):
    runtime.features.clear()
    result = invoke("list_mesh_features", "comp1", "mesh1")
    assert result["success"] and result["data"]["features"] == []


@pytest.mark.parametrize("method", ["getType", "label"])
def test_list_mesh_features_missing_getter(runtime, method):
    delattr(runtime.feature, method)
    result = invoke("list_mesh_features", "comp1", "mesh1")
    assert not result["success"] and f"{method}() getter unavailable" in result["error"]
    runtime.seq.create.assert_not_called()


def test_list_mesh_features_tags_unavailable(runtime):
    runtime.seq.feature.return_value = SimpleNamespace()
    runtime.seq.feature.side_effect = None
    result = invoke("list_mesh_features", "comp1", "mesh1")
    assert not result["success"] and "tags() getter unavailable" in result["error"]


def test_list_mesh_features_java_error_no_retry(runtime):
    runtime.feature.getType.side_effect = RuntimeError("original Java getter error")
    result = invoke("list_mesh_features", "comp1", "mesh1")
    assert not result["success"] and result["error"] == "original Java getter error"
    runtime.feature.getType.assert_called_once_with()
    runtime.seq.create.assert_not_called()


@pytest.fixture
def value_runtime(runtime):
    # Only these read methods exist: any unexpected API fails immediately.
    feature = Mock(spec=["hasProperty", "getValueType", "getString", "getStringArray",
                         "getStringMatrix", "getBoolean", "getInt", "getIntArray",
                         "getDouble", "getDoubleArray", "getDoubleMatrix"])
    feature.hasProperty.return_value = True
    runtime.features["size_cuboid"] = feature
    runtime.value_feature = feature
    return runtime


def read_values(runtime, names):
    return invoke("get_mesh_feature_property_values", "comp1", "mesh1", "size_cuboid", names)


@pytest.mark.parametrize("kind,method,value,representation", [
    ("String", "getString", " c/(acpr.freq*freq)  ", "string"),
    ("StringArray", "getStringArray", ["c/freq", "acpr.k", ""], "string[]"),
    ("StringMatrix", "getStringMatrix", [["c/freq"], []], "string[][]"),
    ("Boolean", "getBoolean", False, "boolean"),
    ("Int", "getInt", 4, "integer"),
    ("IntArray", "getIntArray", [2, 1], "integer[]"),
    ("Double", "getDouble", 1.25, "number"),
    ("DoubleArray", "getDoubleArray", [1.25, 0.0], "number[]"),
    ("DoubleMatrix", "getDoubleMatrix", [[1.25], []], "number[][]"),
    ("DoubleRowMatrix", "getStringArray", ["range(0,c/freq,acpr.k)"], "string[]"),
])
def test_mesh_values_typed_read_only(value_runtime, kind, method, value, representation):
    r = value_runtime
    f = r.value_feature
    f.getValueType.return_value = kind
    getattr(f, method).return_value = value
    result = read_values(r, ["sizeexpr"])
    assert result["success"]
    assert result["data"]["properties"] == [dict(
        name="sizeexpr", value=value, value_type=kind, representation=representation,
        getter=method, available=True, error=None)]
    assert f.mock_calls == [call.hasProperty("sizeexpr"), call.getValueType("sizeexpr"),
                            getattr(call, method)("sizeexpr")]
    assert r.seq.mock_calls == [call.feature(), call.feature("size_cuboid")]
    assert r.selection.mock_calls == []
    assert r.feature.mock_calls == []


def test_mesh_values_multiple_order_missing_and_duplicates(value_runtime):
    f = value_runtime.value_feature
    f.hasProperty.side_effect = lambda n: n != "missing"
    f.getValueType.side_effect = lambda n: "String" if n == "sizeexpr" else "Double"
    f.getString.return_value = " c/(acpr.freq*freq) "
    f.getDouble.return_value = 1.2
    result = read_values(value_runtime, ["sizeexpr", "missing", "hmeshgrad", "sizeexpr"])
    rows = result["data"]["properties"]
    assert result["success"]
    assert [r["name"] for r in rows] == ["sizeexpr", "missing", "hmeshgrad", "sizeexpr"]
    assert rows[0]["value"] == rows[3]["value"] == " c/(acpr.freq*freq) "
    assert rows[2]["value"] == 1.2
    assert not rows[1]["available"] and 'does not exist' in rows[1]["error"]
    assert call.getValueType("missing") not in f.mock_calls


@pytest.mark.parametrize("component,mesh,tag,message", [
    ("absent", "mesh1", "size_cuboid", 'Component "absent" does not exist'),
    ("comp1", "absent", "size_cuboid", 'Mesh "absent" does not exist'),
    ("comp1", "mesh1", "absent", 'Mesh feature "absent" does not exist'),
])
def test_mesh_values_missing_nodes_no_creation(value_runtime, component, mesh, tag, message):
    r = value_runtime
    result = invoke("get_mesh_feature_property_values", component, mesh, tag, ["sizeexpr"])
    assert not result["success"] and message in result["error"]
    assert r.value_feature.mock_calls == []
    r.seq.create.assert_not_called()
    r.seq.automatic.assert_not_called()


@pytest.mark.parametrize("method", ["hasProperty", "getValueType", "getString"])
def test_mesh_values_java_exception_once_no_fallback(value_runtime, method):
    f = value_runtime.value_feature
    f.getValueType.return_value = "String"
    getattr(f, method).side_effect = RuntimeError("original Java getter error")
    result = read_values(value_runtime, ["sizeexpr"])
    row = result["data"]["properties"][0]
    assert not row["available"] and row["error"] == "original Java getter error"
    getattr(f, method).assert_called_once_with("sizeexpr")
    expected = [call.hasProperty("sizeexpr"), call.getValueType("sizeexpr"), call.getString("sizeexpr")]
    assert f.mock_calls == expected[:["hasProperty", "getValueType", "getString"].index(method) + 1]


@pytest.mark.parametrize("failure", ["unknown_type", "null_type", "null_value", "missing_getter"])
def test_mesh_values_unavailable(value_runtime, failure):
    f = value_runtime.value_feature
    f.getValueType.return_value = "String"
    f.getString.return_value = "value"
    if failure == "unknown_type":
        f.getValueType.return_value = "FutureType"
    elif failure == "null_type":
        f.getValueType.return_value = None
    elif failure == "null_value":
        f.getString.return_value = None
    else:
        del f.getString
    result = read_values(value_runtime, ["sizeexpr"])
    row = result["data"]["properties"][0]
    assert not row["available"] and row["value"] is None and row["error"]
    f.getDouble.assert_not_called()
    f.getStringArray.assert_not_called()


def test_mesh_values_empty(value_runtime):
    result = read_values(value_runtime, [])
    assert result["success"] and result["data"]["properties"] == []
    assert value_runtime.value_feature.mock_calls == []
    # Empty requests still validate that the feature exists.
    assert not invoke("get_mesh_feature_property_values", "comp1", "mesh1", "missing", [])["success"]


@pytest.mark.parametrize("names", [None, "sizeexpr", [1], [""], [" "]])
def test_mesh_values_invalid_names(value_runtime, names):
    result = read_values(value_runtime, names)
    assert not result["success"] and "property_names" in result["error"]
    assert value_runtime.value_feature.mock_calls == []

@pytest.fixture
def activity_runtime(runtime, monkeypatch):
    feature = Mock(spec=["active", "isActive"])
    feature.isActive.return_value = True
    runtime.features["size_cuboid"] = feature
    runtime.activity_feature = feature
    monkeypatch.setattr(tools, "_require_model", lambda: runtime.model)
    return runtime


@pytest.mark.parametrize("active", [True, False])
def test_activity_getter_readonly(activity_runtime, monkeypatch, active):
    r = activity_runtime
    r.activity_feature.isActive.return_value = active
    monkeypatch.setattr(tools, "_run_tool", Mock(side_effect=AssertionError("mutation path")))
    monkeypatch.setattr(tools, "_require_visible_main", Mock(side_effect=AssertionError("mutation guard")))
    result = invoke("get_mesh_feature_active", "comp1", "mesh1", "size_cuboid")
    assert result["success"]
    assert result["data"] == dict(component="comp1", mesh="mesh1", feature_tag="size_cuboid",
                                  active=active, available=True, error=None)
    assert r.activity_feature.mock_calls == [call.isActive()]
    assert r.seq.mock_calls == [call.feature(), call.feature("size_cuboid")]


@pytest.mark.parametrize("active", [False, True])
def test_activity_setter_and_readback(activity_runtime, active):
    r = activity_runtime
    r.activity_feature.isActive.return_value = active
    result = invoke("set_mesh_feature_active", "comp1", "mesh1", "size_cuboid", active)
    assert result["success"] and result["data"]["success"]
    assert result["data"]["requested_active"] is active
    assert result["data"]["resulting_active"] is active
    assert result["data"]["error"] is None
    assert r.activity_feature.mock_calls == [call.active(active), call.isActive()]
    assert r.seq.mock_calls == [call.feature(), call.feature("size_cuboid")]
    # Restrictive specs and exact calls prohibit Build/run/save/property changes.


@pytest.mark.parametrize("component,mesh,tag", [
    ("missing", "mesh1", "size_cuboid"), ("comp1", "missing", "size_cuboid"),
    ("comp1", "mesh1", "missing"), ("", "mesh1", "size_cuboid"),
    ("comp1", " ", "size_cuboid"), ("comp1", "mesh1", ""),
])
def test_activity_invalid_tags(activity_runtime, component, mesh, tag):
    for name, extra in [("get_mesh_feature_active", ()), ("set_mesh_feature_active", (False,))]:
        assert not invoke(name, component, mesh, tag, *extra)["success"]
    assert activity_runtime.activity_feature.mock_calls == []


def test_activity_setter_guard(activity_runtime, monkeypatch):
    monkeypatch.setattr(model_module, "_read_workflow_state", lambda: {"visible_main_locked": True, "guard_level": "strict"})
    monkeypatch.setattr(model_module, "_visible_main_mismatch", lambda *a: ["tag mismatch"])
    monkeypatch.setattr(model_module, "_visible_main_identity", lambda *a: {})
    result = invoke("set_mesh_feature_active", "comp1", "mesh1", "size_cuboid", False)
    assert not result["success"] and "lock mismatch" in result["error"]
    assert activity_runtime.activity_feature.mock_calls == []
    assert activity_runtime.seq.mock_calls == []


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_activity_bool_validation(activity_runtime, value):
    result = invoke("set_mesh_feature_active", "comp1", "mesh1", "size_cuboid", value)
    assert not result["success"] and "bool" in result["error"]
    assert activity_runtime.activity_feature.mock_calls == []


@pytest.mark.parametrize("failure", ["missing", "null", "exception", "mismatch"])
def test_activity_readback_failure(activity_runtime, failure):
    f = activity_runtime.activity_feature
    if failure == "missing":
        del f.isActive
    elif failure == "null":
        f.isActive.return_value = None
    elif failure == "exception":
        f.isActive.side_effect = RuntimeError("Java getter error")
    else:
        f.isActive.return_value = True
    result = invoke("set_mesh_feature_active", "comp1", "mesh1", "size_cuboid", False)
    assert not result["data"]["success"] and result["data"]["error"]
    f.active.assert_called_once_with(False)
    if failure != "missing":
        f.isActive.assert_called_once_with()
    if failure != "mismatch":
        f.reset_mock()
        result = invoke("get_mesh_feature_active", "comp1", "mesh1", "size_cuboid")
        assert result["data"]["active"] is None
        assert not result["data"]["available"] and result["data"]["error"]
        f.active.assert_not_called()


def test_activity_setter_error_no_retry(activity_runtime):
    f = activity_runtime.activity_feature
    f.active.side_effect = RuntimeError("Java setter error")
    result = invoke("set_mesh_feature_active", "comp1", "mesh1", "size_cuboid", False)
    assert not result["success"] and result["error"] == "Java setter error"
    assert f.mock_calls == [call.active(False)]


@pytest.fixture
def move_runtime(runtime):
    # Restrictive specs prohibit builds, saves, property edits and creation.
    order = ["size", "se1", "ftet1", "size_local"]
    collection = Mock(spec=["tags", "move"])
    collection.tags.side_effect = lambda: order.copy()

    def move(tag, position):
        order.remove(tag)
        order.insert(position, tag)

    collection.move.side_effect = move
    runtime.seq.feature.side_effect = lambda: collection
    runtime.collection = collection
    runtime.order = order
    return runtime


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_move_exact_java_call_and_order_no_build(move_runtime, position):
    r = move_runtime
    result = invoke("move_mesh_feature", "comp1", "mesh1", "size_local", position)
    assert result["success"]
    assert result["data"] == dict(component="comp1", mesh="mesh1", feature_tag="size_local",
        requested_position=position, resulting_position=position,
        ordered_features=r.order, success=True, error=None)
    assert r.collection.mock_calls == [call.tags(), call.move("size_local", position), call.tags()]
    assert r.seq.mock_calls == [call.feature()]
    assert r.feature.mock_calls == []
    assert r.selection.mock_calls == []


@pytest.mark.parametrize("component,mesh,tag", [
    ("missing", "mesh1", "size_local"), ("comp1", "missing", "size_local"),
    ("comp1", "mesh1", "missing"), ("", "mesh1", "size_local"),
    ("comp1", " ", "size_local"), ("comp1", "mesh1", ""),
])
def test_move_invalid_tags(move_runtime, component, mesh, tag):
    result = invoke("move_mesh_feature", component, mesh, tag, 0)
    assert not result["success"] and result["error"]
    move_runtime.collection.move.assert_not_called()


@pytest.mark.parametrize("position", [-1, 4, True, 1.0, "0", None])
def test_move_invalid_position(move_runtime, position):
    result = invoke("move_mesh_feature", "comp1", "mesh1", "size_local", position)
    assert not result["success"] and "position" in result["error"]
    move_runtime.collection.move.assert_not_called()


def test_move_requires_visible_main_guard(move_runtime, monkeypatch):
    monkeypatch.setattr(model_module, "_read_workflow_state",
        lambda: {"visible_main_locked": True, "guard_level": "strict"})
    monkeypatch.setattr(model_module, "_visible_main_mismatch", lambda *a: ["tag mismatch"])
    monkeypatch.setattr(model_module, "_visible_main_identity", lambda *a: {})
    result = invoke("move_mesh_feature", "comp1", "mesh1", "size_local", 0)
    assert not result["success"] and "lock mismatch" in result["error"]
    assert move_runtime.seq.mock_calls == []
    assert move_runtime.collection.mock_calls == []


def test_move_java_error_no_retry(move_runtime):
    move_runtime.collection.move.side_effect = RuntimeError("Java move error")
    result = invoke("move_mesh_feature", "comp1", "mesh1", "size_local", 0)
    assert not result["success"] and result["error"] == "Java move error"
    assert move_runtime.collection.mock_calls == [call.tags(), call.move("size_local", 0)]


@pytest.mark.parametrize("failure", ["exception", "null", "mismatch"])
def test_move_readback_failure_no_retry(move_runtime, failure):
    r = move_runtime
    after = RuntimeError("Java tags error") if failure == "exception" else (
        None if failure == "null" else r.order.copy())
    r.collection.tags.side_effect = [r.order.copy(), after]
    result = invoke("move_mesh_feature", "comp1", "mesh1", "size_local", 0)
    assert result["success"]  # Wrapper accepted execution; payload reports verification failure.
    assert not result["data"]["success"] and result["data"]["error"]
    r.collection.move.assert_called_once_with("size_local", 0)


def test_move_public_signature():
    import inspect
    signature = inspect.signature(tools.move_mesh_feature)
    assert list(signature.parameters) == ["component", "mesh", "feature_tag", "position"]
    assert all(p.default is inspect.Parameter.empty for p in signature.parameters.values())
    assert signature.return_annotation == "str"
