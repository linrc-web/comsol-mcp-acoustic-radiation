# Strict read-only global parameter evaluation

`evaluate_parameter(parameter_name: str) -> str` returns the usual MCP JSON
envelope. Its `data` contains `parameter_name`, `value`, `unit`, `getter`,
`unit_getter`, `available`, `error`, and `unit_error`.

## Local COMSOL 6.4 API evidence

Verified from interface class metadata in the official installation file
`D:\Program Files\COMSOL\COMSOL64\Multiphysics\plugins\com.comsol.api_1.0.0.jar`:

```java
// com.comsol.model.Model
public abstract ModelParam param();

// ModelParam extends ParamBase; ParamBase extends ExpressionBase
// com.comsol.model.ParamBase
public abstract double evaluate(java.lang.String);
public abstract double evaluate(java.lang.String, java.lang.String);
public abstract java.lang.String evaluateUnit(java.lang.String);
```

The one-argument evaluation descriptor is `(Ljava/lang/String;)D` and the unit
getter descriptor is `(Ljava/lang/String;)Ljava/lang/String;`.

The tool uses only `model.java.param()`, `evaluate(parameter_name)`, and
`evaluateUnit(parameter_name)` for parameter access. The parameter name is passed
unchanged. It never calls MPh `model.evaluate`, creates evaluation/result/dataset
nodes, or builds, solves, saves, or sets parameters. It is registered as SAFE_READ
and uses the existing read-only wrapper and visible-main guard.

Evaluation failure returns `available=false`, `value=null`, and the actual error
text, without a fallback or unit query. Successful evaluation returns
`available=true`. If the unit getter is missing, throws, or returns null, the value
is retained, `unit=null`, and `unit_error` explains why. Empty unit strings are
preserved exactly. Units are never inferred or parsed from parameter expressions.

Validation uses strict mocks only; no live server or model is used. Changes to
source registration do not update an already running MCP process automatically.
