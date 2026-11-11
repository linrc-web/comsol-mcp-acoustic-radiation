"""Geometry object selections: mock-only, with no COMSOL/JVM connection."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import comsol_mcp._state as state
import comsol_mcp._tools_geometry as tools


@pytest.fixture
def runtime(monkeypatch):
    model = Mock()
    geom = model.java.component.return_value.geom.return_value
    collection = Mock()
    collection.tags.return_value = ["rot1"]
    feature = Mock()
    selection = feature.selection.return_value
    selection.objects.return_value = ["cuboid1"]
    geom.feature.side_effect = lambda tag=None: collection if tag is None else feature
    monkeypatch.setattr(tools, "_require_visible_main", Mock(return_value=model))
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(state, "_tool_result", lambda tool, success, data=None, error="":
                        json.dumps(dict(tool=tool, success=success, data=data, error=error)))
    return SimpleNamespace(model=model, geom=geom, collection=collection,
                           feature=feature, selection=selection)


def invoke(setter=False, **kwargs):
    args = dict(component="comp1", geometry="geom1", feature_tag="rot1",
                selection_name="input")
    args.update(kwargs)
    fn = tools.set_geometry_feature_selection if setter else tools.get_geometry_feature_selection
    return json.loads(fn(**args))


@pytest.mark.parametrize("tags", [["cuboid1"], ["obj1", "obj2"]])
def test_set_objects(runtime, tags):
    result = invoke(True, object_tags=tags)
    assert result["success"] and result["data"]["object_tags"] == tags
    runtime.feature.selection.assert_called_once_with("input")
    runtime.selection.set.assert_called_once_with(tags)
    runtime.geom.run.assert_not_called()
    runtime.collection.create.assert_not_called()


@pytest.mark.parametrize("tags", [[], ["cuboid1"], ["obj1", "obj2"]])
def test_get_objects_is_strictly_readonly(runtime, tags):
    runtime.selection.objects.return_value = tags
    result = invoke()
    assert result["success"] and result["data"]["object_tags"] == tags
    # Exhaustively allow only navigation/read calls on the Java model tree.
    assert runtime.model.mock_calls == [
        call.java.component("comp1"),
        call.java.component().geom("geom1"),
        call.java.component().geom().feature(),
        call.java.component().geom().feature("rot1"),
    ]
    assert runtime.collection.mock_calls == [call.tags()]
    assert runtime.feature.mock_calls == [call.selection("input"),
                                          call.selection().objects()]


@pytest.mark.parametrize("setter", [False, True])
def test_missing_feature(runtime, setter):
    runtime.collection.tags.return_value = []
    result = invoke(setter, **({"object_tags": ["cuboid1"]} if setter else {}))
    assert not result["success"]
    assert result["error"] == 'Geometry feature "rot1" does not exist.'
    runtime.feature.selection.assert_not_called()
    runtime.collection.create.assert_not_called()


@pytest.mark.parametrize("setter", [False, True])
def test_missing_selection_propagates_without_retry(runtime, setter):
    runtime.feature.selection.side_effect = RuntimeError("Unknown selection: missing")
    result = invoke(setter, selection_name="missing",
                    **({"object_tags": ["cuboid1"]} if setter else {}))
    assert not result["success"] and result["error"] == "Unknown selection: missing"
    runtime.feature.selection.assert_called_once_with("missing")
    runtime.selection.set.assert_not_called()
    runtime.collection.create.assert_not_called()


@pytest.mark.parametrize("tags", [None, "cuboid1", [], [""], [" "], [1], ["obj1", None]])
def test_invalid_object_tags_before_model_access(runtime, tags):
    result = invoke(True, object_tags=tags)
    assert not result["success"] and "object_tags" in result["error"]
    runtime.model.java.component.assert_not_called()


@pytest.mark.parametrize("field", ["component", "geometry", "feature_tag", "selection_name"])
def test_empty_path_rejected_without_java_access(runtime, field):
    result = invoke(**{field: " "})
    assert not result["success"] and field in result["error"]
    runtime.model.java.component.assert_not_called()


@pytest.mark.parametrize("setter", [False, True])
def test_java_failure_propagates_without_retry(runtime, setter):
    method = runtime.selection.set if setter else runtime.selection.objects
    method.side_effect = RuntimeError("Java selection failure")
    result = invoke(setter, **({"object_tags": ["cuboid1"]} if setter else {}))
    assert not result["success"] and result["error"] == "Java selection failure"
    assert method.call_count == 1


def test_null_read_is_error_not_empty_selection(runtime):
    runtime.selection.objects.return_value = None
    result = invoke()
    assert not result["success"] and "returned null" in result["error"]


def test_generic_selection_name_and_feature(runtime):
    runtime.collection.tags.return_value = ["dif1"]
    result = invoke(True, feature_tag="dif1", selection_name="input2",
                    object_tags=["obj1", "obj2"])
    assert result["success"]
    runtime.geom.feature.assert_any_call("dif1")
    runtime.feature.selection.assert_called_once_with("input2")


def test_tools_use_correct_execution_wrapper(runtime, monkeypatch):
    read = Mock(wraps=tools._run_tool_readonly)
    write = Mock(wraps=tools._run_tool)
    monkeypatch.setattr(tools, "_run_tool_readonly", read)
    monkeypatch.setattr(tools, "_run_tool", write)
    assert invoke()["success"]
    read.assert_called_once()
    write.assert_not_called()
    assert invoke(True, object_tags=["cuboid1"])["success"]
    write.assert_called_once()
