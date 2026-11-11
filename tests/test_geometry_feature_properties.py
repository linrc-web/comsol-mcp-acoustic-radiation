"""Mock-only property reads: no JVM, server connection, or model mutation."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import comsol_mcp._state as state
import comsol_mcp._tools_geometry as tools


@pytest.fixture
def runtime(monkeypatch):
    # Exhaustive call assertions below reject every non-navigation/non-getter call.
    model = Mock()
    geom = model.java.component.return_value.geom.return_value
    collection = Mock()
    collection.tags.return_value = ["feature1"]
    feature = Mock()
    feature.hasProperty.return_value = True
    feature.getValueType.return_value = "String"
    feature.getString.return_value = "theta"
    geom.feature.side_effect = lambda tag=None: collection if tag is None else feature
    require = Mock(return_value=model)
    monkeypatch.setattr(tools, "_require_visible_main", require)
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error)))
    return SimpleNamespace(model=model, geom=geom, collection=collection,
                           feature=feature, require=require)


def invoke(property_names=None, **kwargs):
    args = dict(component="comp1", geometry="geom1", feature_tag="feature1",
                property_names=["angle"] if property_names is None else property_names)
    args.update(kwargs)
    return json.loads(tools.get_geometry_feature_properties(**args))


@pytest.mark.parametrize("value_type,getter,value", [
    ("String", "getString", "theta"),
    ("StringArray", "getStringArray", ["xc", "yc", "zc"]),
    ("StringMatrix", "getStringMatrix", [["xc", "theta"], ["yc", "zc"]]),
    ("Boolean", "getBoolean", True),
    ("BooleanArray", "getBooleanArray", [True, False]),
    ("BooleanMatrix", "getBooleanMatrix", [[True, False]]),
    ("Int", "getInt", 3),
    ("IntArray", "getIntArray", [1, 2]),
    ("IntMatrix", "getIntMatrix", [[1, 2]]),
    ("Double", "getDouble", 1.25),
    ("DoubleArray", "getDoubleArray", [0., 0., 1.]),
    ("DoubleMatrix", "getDoubleMatrix", [[1., 2.]]),
    ("DoubleRowMatrix", "getStringArray", ["range(0,theta,90)", "xc"]),
])
def test_typed_read_is_strictly_readonly(runtime, value_type, getter, value):
    runtime.feature.getValueType.return_value = value_type
    getattr(runtime.feature, getter).return_value = value
    result = invoke()
    assert result["success"]
    assert result["data"]["properties"] == [dict(
        name="angle", value_type=value_type, getter=getter, value=value)]
    assert runtime.model.mock_calls == [
        call.java.component("comp1"), call.java.component().geom("geom1"),
        call.java.component().geom().feature(),
        call.java.component().geom().feature("feature1"),
    ]
    assert runtime.collection.mock_calls == [call.tags()]
    assert runtime.feature.mock_calls == [
        call.hasProperty("angle"), call.getValueType("angle"),
        getattr(call, getter)("angle"),
    ]


def test_multiple_properties_and_no_setter_cache(runtime):
    runtime.feature.getValueType.side_effect = ["String", "StringArray", "String"]
    runtime.feature.getString.side_effect = ["theta", "z"]
    runtime.feature.getStringArray.return_value = ["xc", "yc", "zc"]
    result = invoke(["angle", "center", "axis"])
    assert result["success"]
    assert [p["value"] for p in result["data"]["properties"]] == [
        "theta", ["xc", "yc", "zc"], "z"]
    # A later read must query COMSOL again, rather than reuse prior input/output.
    runtime.feature.getValueType.side_effect = None
    runtime.feature.getValueType.return_value = "String"
    runtime.feature.getString.side_effect = None
    runtime.feature.getString.return_value = "2*theta"
    assert invoke()["data"]["properties"][0]["value"] == "2*theta"


def test_missing_feature(runtime):
    runtime.collection.tags.return_value = []
    result = invoke()
    assert not result["success"]
    assert result["error"] == 'Geometry feature "feature1" does not exist.'
    assert runtime.geom.feature.call_args_list == [call()]
    assert runtime.feature.mock_calls == []


def test_missing_property_stops(runtime):
    runtime.feature.hasProperty.return_value = False
    result = invoke(["missing", "angle"])
    assert not result["success"]
    assert result["error"] == 'Geometry feature "feature1" has no property "missing".'
    assert runtime.feature.mock_calls == [call.hasProperty("missing")]


def test_unknown_type_never_guesses(runtime):
    runtime.feature.getValueType.return_value = "UnknownType"
    result = invoke()
    assert not result["success"] and '"UnknownType"' in result["error"]
    assert runtime.feature.mock_calls == [call.hasProperty("angle"), call.getValueType("angle")]


@pytest.mark.parametrize("method", ["hasProperty", "getValueType", "getString"])
def test_java_error_propagates_without_retry(runtime, method):
    getattr(runtime.feature, method).side_effect = RuntimeError("Original Java error")
    result = invoke(["angle", "next"])
    assert not result["success"] and result["error"] == "Original Java error"
    getattr(runtime.feature, method).assert_called_once_with("angle")
    assert all(c.args != ("next",) for c in runtime.feature.mock_calls)


@pytest.mark.parametrize("field", ["component", "geometry", "feature_tag"])
def test_invalid_path_no_java_access(runtime, field):
    result = invoke(**{field: " "})
    assert not result["success"] and field in result["error"]
    assert runtime.model.mock_calls == []


@pytest.mark.parametrize("names", [[], "angle", [""], [" "], [None], [1]])
def test_invalid_names_no_java_access(runtime, names):
    result = invoke(names)
    assert not result["success"] and "property_names" in result["error"]
    assert runtime.model.mock_calls == []


def test_null_names_no_java_access(runtime):
    result = json.loads(tools.get_geometry_feature_properties("comp1", "geom1", "feature1", None))
    assert not result["success"] and "property_names" in result["error"]
    assert runtime.model.mock_calls == []


def test_null_value_preserved(runtime):
    runtime.feature.getString.return_value = None
    assert invoke()["data"]["properties"][0]["value"] is None


def test_readonly_wrapper_and_guard(runtime, monkeypatch):
    read = Mock(wraps=tools._run_tool_readonly)
    write = Mock(wraps=tools._run_tool)
    monkeypatch.setattr(tools, "_run_tool_readonly", read)
    monkeypatch.setattr(tools, "_run_tool", write)
    assert invoke()["success"]
    read.assert_called_once()
    write.assert_not_called()
    runtime.require.assert_called_once_with("get_geometry_feature_properties")


@pytest.mark.parametrize("typed,strings", [
    ([29.999999999999996], ["theta"]),
    ([29.999999999999996], ["30"]),
    ([0.0001875, 0., 0.], ["xc", "yc", "zc"]),
])
def test_optional_string_probe_preserves_both_values(runtime, typed, strings):
    runtime.feature.getValueType.return_value = "DoubleArray"
    runtime.feature.getDoubleArray.return_value = typed
    runtime.feature.getStringArray.return_value = strings
    result = invoke(include_string_representation=True)
    assert result["success"]
    assert result["data"]["properties"] == [dict(
        name="angle", value_type="DoubleArray", getter="getDoubleArray", value=typed,
        string_getter="getStringArray", string_value=strings,
        string_available=True, string_error=None)]
    assert runtime.feature.mock_calls == [
        call.hasProperty("angle"), call.getValueType("angle"),
        call.getDoubleArray("angle"), call.getStringArray("angle")]


def test_string_probe_error_preserves_typed_value_without_mutation(runtime):
    runtime.feature.getValueType.return_value = "DoubleArray"
    runtime.feature.getDoubleArray.return_value = [30.]
    runtime.feature.getStringArray.side_effect = RuntimeError("Original string getter error")
    result = invoke(["rot", "pos"], include_string_representation=True)
    assert result["success"]
    for item in result["data"]["properties"]:
        assert item["value"] == [30.]
        assert item["string_value"] is None
        assert item["string_available"] is False
        assert item["string_error"] == "Original string getter error"
    assert runtime.feature.mock_calls == [c for name in ["rot", "pos"] for c in [
        call.hasProperty(name), call.getValueType(name),
        call.getDoubleArray(name), call.getStringArray(name)]]


def test_explicit_false_keeps_old_double_array_result(runtime):
    runtime.feature.getValueType.return_value = "DoubleArray"
    runtime.feature.getDoubleArray.return_value = [30.]
    assert invoke(include_string_representation=False)["data"]["properties"] == [dict(
        name="angle", value_type="DoubleArray", getter="getDoubleArray", value=[30.])]
    runtime.feature.getStringArray.assert_not_called()


def test_scalar_probe_is_unavailable_without_extra_java_calls(runtime):
    item = invoke(include_string_representation=True)["data"]["properties"][0]
    assert item["value"] == "theta"
    assert item["string_getter"] is None
    assert item["string_available"] is False
    assert "vector" in item["string_error"]
    runtime.feature.getStringArray.assert_not_called()


def test_null_string_probe_is_unavailable(runtime):
    runtime.feature.getValueType.return_value = "DoubleArray"
    runtime.feature.getDoubleArray.return_value = [30.]
    runtime.feature.getStringArray.return_value = None
    item = invoke(include_string_representation=True)["data"]["properties"][0]
    assert item["value"] == [30.]
    assert item["string_available"] is False
    assert item["string_error"] == "getStringArray returned null."


def test_string_probe_retains_safe_read_classification():
    from comsol_mcp._server import SAFE_READ_TOOLS
    assert "get_geometry_feature_properties" in SAFE_READ_TOOLS
