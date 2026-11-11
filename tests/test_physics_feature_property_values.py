"""Strict read-only physics value reads: mocks only, no JVM or server."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import comsol_mcp._state as state
import comsol_mcp._tools_physics as tools
from comsol_mcp._server import SAFE_READ_TOOLS, SAFE_VISIBLE_MAIN_WRITE_TOOLS, RESTRICTED_TOOLS


@pytest.fixture
def runtime(monkeypatch):
    feature = Mock(spec=["hasProperty", "getValueType", "getString", "getStringArray",
                         "getStringMatrix", "getBoolean", "getInt", "getIntArray",
                         "getDouble", "getDoubleArray", "getDoubleMatrix"])
    feature.hasProperty.return_value = True
    features = Mock(spec=["tags"])
    features.tags.return_value = ["fpam1"]
    physics = Mock(spec=["feature"])
    physics.feature.side_effect = lambda tag=None: features if tag is None else feature
    physics_list = Mock(spec=["tags"])
    physics_list.tags.return_value = ["acpr"]
    comp = Mock(spec=["physics"])
    comp.physics.side_effect = lambda tag=None: physics_list if tag is None else physics
    components = Mock(spec=["tags"])
    components.tags.return_value = ["comp1"]
    java = Mock(spec=["component"])
    java.component.side_effect = lambda tag=None: components if tag is None else comp
    model = SimpleNamespace(java=java)
    require = Mock(return_value=model)
    monkeypatch.setattr(tools, "_require_visible_main", require)
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error)))
    monkeypatch.setattr(tools, "_run_tool", Mock(side_effect=AssertionError("mutation wrapper")))
    return SimpleNamespace(**locals())


def invoke(names, component="comp1", physics_tag="acpr", feature_tag="fpam1"):
    return json.loads(tools.get_physics_feature_property_values(
        component, physics_tag, feature_tag, names))


@pytest.mark.parametrize("kind,getter,value,representation", [
    ("String", "getString", " c0 + 0*freq  ", "string"),
    ("StringArray", "getStringArray", ["c0", ""], "string[]"),
    ("StringMatrix", "getStringMatrix", [["c0"], []], "string[][]"),
    ("Boolean", "getBoolean", False, "boolean"),
    ("Int", "getInt", 0, "integer"),
    ("IntArray", "getIntArray", [2, 0], "integer[]"),
    ("Double", "getDouble", 0.0, "number"),
    ("DoubleArray", "getDoubleArray", [1.25, 0.0], "number[]"),
    ("DoubleMatrix", "getDoubleMatrix", [[1.25], []], "number[][]"),
    ("DoubleRowMatrix", "getStringArray", ["range(0,c0,10)"], "string[]"),
])
def test_typed_values_and_exact_readonly_calls(runtime, kind, getter, value, representation):
    r = runtime
    r.feature.getValueType.return_value = kind
    getattr(r.feature, getter).return_value = value
    result = invoke(["c"])
    assert result["success"]
    assert result["data"] == dict(component="comp1", physics_tag="acpr", feature_tag="fpam1",
        properties=[dict(name="c", value=value, value_type=kind, representation=representation,
                         getter=getter, available=True, error=None)])
    assert r.feature.mock_calls == [call.hasProperty("c"), call.getValueType("c"),
                                    getattr(call, getter)("c")]
    assert r.java.mock_calls == [call.component(), call.component("comp1")]
    assert r.components.mock_calls == [call.tags()]
    assert r.comp.mock_calls == [call.physics(), call.physics("acpr")]
    assert r.physics_list.mock_calls == [call.tags()]
    assert r.physics.mock_calls == [call.feature(), call.feature("fpam1")]
    assert r.features.mock_calls == [call.tags()]


@pytest.mark.parametrize("failure", ["missing", "unsupported", "None", "null_type",
                                    "null_value", "missing_getter", "hasProperty",
                                    "getValueType", "getString"])
def test_property_failures_preserve_other_values_without_fallback(runtime, failure):
    f = runtime.feature
    f.getValueType.return_value = "String"
    f.getString.return_value = "c0"
    if failure == "missing":
        f.hasProperty.side_effect = lambda n: n != "bad"
    elif failure in ("unsupported", "None", "null_type"):
        kind = {"unsupported": "FutureType", "None": "None", "null_type": None}[failure]
        f.getValueType.side_effect = lambda n: kind if n == "bad" else "String"
    elif failure == "null_value":
        f.getString.side_effect = lambda n: None if n == "bad" else "c0"
    elif failure == "missing_getter":
        f.getValueType.side_effect = lambda n: "Double" if n == "bad" else "String"
        del f.getDouble
    else:
        def fail_bad(n):
            if n == "bad":
                raise RuntimeError("original Java getter error")
            return {"hasProperty": True, "getValueType": "String", "getString": "c0"}[failure]
        getattr(f, failure).side_effect = fail_bad
    result = invoke(["c", "bad", "c"])
    assert result["success"]
    rows = result["data"]["properties"]
    assert [row["name"] for row in rows] == ["c", "bad", "c"]
    assert rows[0] == rows[2] and rows[0]["value"] == "c0"
    assert not rows[1]["available"] and rows[1]["value"] is None and rows[1]["error"]
    if failure in ("hasProperty", "getValueType", "getString"):
        assert rows[1]["error"] == "original Java getter error"
    expected = [call.hasProperty("bad"), call.getValueType("bad"), call.getString("bad")]
    stop = {"missing": 1, "hasProperty": 1, "getValueType": 2,
            "unsupported": 2, "None": 2, "null_type": 2, "missing_getter": 2}.get(failure, 3)
    assert [c for c in f.mock_calls if c.args == ("bad",)] == expected[:stop]


@pytest.mark.parametrize("scope,message", [("components", 'Component "comp1"'),
    ("physics_list", 'Physics "acpr"'), ("features", 'Physics feature "fpam1"')])
def test_missing_nodes_fail_without_creation_even_for_empty_request(runtime, scope, message):
    getattr(runtime, scope).tags.return_value = []
    result = invoke([])
    assert not result["success"] and message in result["error"]
    assert "does not exist" in result["error"]
    assert runtime.feature.mock_calls == []
    if scope == "components":
        assert runtime.java.mock_calls == [call.component()]
    elif scope == "physics_list":
        assert runtime.comp.mock_calls == [call.physics()]
    else:
        assert runtime.physics.mock_calls == [call.feature()]


def test_empty_request(runtime):
    assert invoke([])["data"]["properties"] == []
    assert runtime.feature.mock_calls == []
    runtime.physics.feature.assert_any_call("fpam1")


@pytest.mark.parametrize("names", [None, "c", [1], [""], [" "]])
def test_invalid_property_names(runtime, names):
    result = invoke(names)
    assert not result["success"] and "property_names" in result["error"]
    assert runtime.feature.mock_calls == []


@pytest.mark.parametrize("tags", [("", "acpr", "fpam1"), ("comp1", " ", "fpam1"),
                                  ("comp1", "acpr", None)])
def test_invalid_tags_before_model_access(runtime, tags):
    assert not invoke([], *tags)["success"]
    runtime.require.assert_not_called()


def test_readonly_classification_and_registration(runtime, monkeypatch):
    reader = Mock(wraps=tools._run_tool_readonly)
    monkeypatch.setattr(tools, "_run_tool_readonly", reader)
    assert invoke([])["success"]
    reader.assert_called_once()
    tools._run_tool.assert_not_called()
    name = "get_physics_feature_property_values"
    assert name in SAFE_READ_TOOLS
    assert name not in SAFE_VISIBLE_MAIN_WRITE_TOOLS | RESTRICTED_TOOLS
    registry = Mock()
    tools.register(registry)
    registry.add_tool.assert_any_call(tools.get_physics_feature_property_values)
    import comsol_mcp.mcp_server
    from comsol_mcp._server import mcp
    registered = mcp._tool_manager._tools
    assert name in registered
    assert "get_physics_selection" in registered and "list_physics_features" in registered
