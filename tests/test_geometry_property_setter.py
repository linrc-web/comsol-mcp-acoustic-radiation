"""Setter dispatch tests using only a fake feature; no JVM or server."""

import json
from unittest.mock import Mock, call

import pytest

from comsol_mcp._model_ops import _apply_feature_properties, _normalize_properties


@pytest.mark.parametrize("entry, expected", [
    ({"name": "rot", "value": "theta"}, "theta"),
    ({"name": "rot", "values": ["theta"]}, ["theta"]),
    ({"name": "pos", "values": ["xc", "yc", "zc"]}, ["xc", "yc", "zc"]),
    ({"name": "keep", "value": "on"}, "on"),
    ({"name": "custom", "values": ["a"]}, ["a"]),
    ({"name": "size", "values": [1, 2, 3]}, ["1", "2", "3"]),
    ({"name": "number", "value": 42}, "42"),
    ({"name": "flag", "value": True}, "True"),
    ({"name": "empty", "values": []}, []),
])
def test_explicit_value_shape(entry, expected):
    feature = Mock(spec=["set"])
    normalized = _normalize_properties(json.dumps([entry]))
    assert normalized == [(entry["name"], expected)]
    result = _apply_feature_properties(feature, normalized)
    # Exhaustive call list: exactly one setter and no additional operations.
    assert feature.mock_calls == [call.set(entry["name"], expected)]
    key = "value" if isinstance(expected, str) else "values"
    assert result == [{"name": entry["name"], key: expected}]


@pytest.mark.parametrize("value", ["theta", ["theta"], ["xc", "yc", "zc"], []])
def test_object_shorthand_preserves_shape(value):
    feature = Mock(spec=["set"])
    _apply_feature_properties(feature, _normalize_properties(json.dumps({"custom": value})))
    assert feature.mock_calls == [call.set("custom", value)]


def test_setter_error_propagates_without_retry():
    feature = Mock(spec=["set"])
    feature.set.side_effect = RuntimeError("setter failed")
    with pytest.raises(RuntimeError, match="setter failed"):
        _apply_feature_properties(feature, _normalize_properties('{"custom":["theta"]}'))
    assert feature.mock_calls == [call.set("custom", ["theta"])]
