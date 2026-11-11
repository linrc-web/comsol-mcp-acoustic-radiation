"""Global parameter evaluation: mocks only, no JVM, server or model mutations."""

import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import comsol_mcp._state as state
import comsol_mcp._tools_params as tools
from comsol_mcp._server import (
    RESTRICTED_TOOLS, SAFE_READ_TOOLS, SAFE_VISIBLE_MAIN_WRITE_TOOLS,
)


@pytest.fixture
def runtime(monkeypatch):
    # These strict mocks expose only the allowed Java getters. Any result(),
    # dataset(), setter, or MPh model.evaluate() access fails the test.
    parameters = Mock(spec_set=["evaluate", "evaluateUnit"])
    parameters.evaluate.return_value = 343.0
    parameters.evaluateUnit.return_value = "m/s"
    java = Mock(spec_set=["param"])
    java.param.return_value = parameters
    model = Mock(spec_set=["java"])
    model.java = java
    require = Mock(return_value=model)
    monkeypatch.setattr(tools, "_require_visible_main", require)
    monkeypatch.setattr(state, "_setup_logging", lambda: None)
    monkeypatch.setattr(
        state, "_tool_result",
        lambda tool, success, data=None, error="": json.dumps(
            dict(tool=tool, success=success, data=data, error=error)),
    )
    mutation_wrapper = Mock(side_effect=AssertionError("mutation wrapper"))
    monkeypatch.setattr(tools, "_run_tool", mutation_wrapper)
    return SimpleNamespace(**locals())


def invoke(name="c0"):
    return json.loads(tools.evaluate_parameter(name))


def test_parameter_value_unit_and_only_allowed_getters(runtime):
    result = invoke()
    assert result["success"]
    assert result["data"] == {
        "parameter_name": "c0", "value": 343.0, "unit": "m/s",
        "getter": "param().evaluate", "unit_getter": "param().evaluateUnit",
        "available": True, "error": None, "unit_error": None,
    }
    runtime.require.assert_called_once_with("evaluate_parameter")
    assert runtime.java.mock_calls == [
        call.param(), call.param().evaluate("c0"), call.param().evaluateUnit("c0"),
    ]
    assert runtime.parameters.mock_calls == [call.evaluate("c0"), call.evaluateUnit("c0")]
    runtime.mutation_wrapper.assert_not_called()


@pytest.mark.parametrize("name,message", [
    ("missing", "Unknown parameter: missing"),
    ("c0", "com.comsol.util.exceptions.FlException: evaluation failed"),
])
def test_missing_parameter_and_java_exception_no_fallback(runtime, name, message):
    runtime.parameters.evaluate.side_effect = RuntimeError(message)
    result = invoke(name)
    assert result["success"]
    row = result["data"]
    assert row["parameter_name"] == name
    assert row["value"] is None and row["unit"] is None
    assert row["available"] is False and row["error"] == message
    assert runtime.parameters.mock_calls == [call.evaluate(name)]
    runtime.mutation_wrapper.assert_not_called()


@pytest.mark.parametrize("failure", ["null", "missing_getter", "java_exception"])
def test_unit_unavailable_preserves_value_and_reports_reason(runtime, failure):
    if failure == "null":
        runtime.parameters.evaluateUnit.return_value = None
    elif failure == "missing_getter":
        del runtime.parameters.evaluateUnit
    else:
        runtime.parameters.evaluateUnit.side_effect = RuntimeError("actual Java unit error")
    row = invoke()["data"]
    assert row["available"] is True and row["value"] == 343.0
    assert row["unit"] is None and row["error"] is None
    assert row["unit_error"]
    if failure == "java_exception":
        assert row["unit_error"] == "actual Java unit error"
    expected = [call.evaluate("c0")]
    if failure != "missing_getter":
        expected.append(call.evaluateUnit("c0"))
    assert runtime.parameters.mock_calls == expected


def test_dimensionless_unit_is_preserved_without_guessing(runtime):
    runtime.parameters.evaluate.return_value = 0.0
    runtime.parameters.evaluateUnit.return_value = ""
    row = invoke("ratio")["data"]
    assert row["available"] is True and row["value"] == 0.0
    assert row["unit"] == "" and row["unit_error"] is None


@pytest.mark.parametrize("name", [None, 1, "", "  "])
def test_invalid_name_fails_before_model_access(runtime, name):
    result = invoke(name)
    assert not result["success"] and "parameter_name" in result["error"]
    runtime.require.assert_not_called()
    assert runtime.java.mock_calls == []


def test_exact_parameter_name_is_passed_without_rewriting(runtime):
    row = invoke(" c0 ")["data"]
    assert row["parameter_name"] == " c0 "
    assert runtime.parameters.mock_calls == [call.evaluate(" c0 "), call.evaluateUnit(" c0 ")]


def test_readonly_wrapper_classification_registration_and_signature(runtime, monkeypatch):
    reader = Mock(wraps=tools._run_tool_readonly)
    monkeypatch.setattr(tools, "_run_tool_readonly", reader)
    assert invoke()["data"]["available"]
    reader.assert_called_once()
    assert reader.call_args.args[0] == "evaluate_parameter"
    runtime.mutation_wrapper.assert_not_called()
    name = "evaluate_parameter"
    assert name in SAFE_READ_TOOLS
    assert name not in SAFE_VISIBLE_MAIN_WRITE_TOOLS | RESTRICTED_TOOLS
    registry = Mock()
    tools.register(registry)
    registry.add_tool.assert_any_call(tools.evaluate_parameter)
    import comsol_mcp.mcp_server
    from comsol_mcp._server import mcp
    assert name in mcp._tool_manager._tools
    signature = inspect.signature(tools.evaluate_parameter)
    assert list(signature.parameters) == ["parameter_name"]
    assert signature.parameters["parameter_name"].annotation == "str"
    assert signature.return_annotation == "str"
