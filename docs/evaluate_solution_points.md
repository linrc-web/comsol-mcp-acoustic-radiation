# Guarded solution point interpolation

`evaluate_solution_points(expressions, coordinates, dataset, solution_tag,
component, geometry, domain_ids, solnum, expected_frequency_hz,
coordinate_unit="m", outersolnum=None)` uses native list inputs.

This tool is classified as `SAFE_VISIBLE_MAIN_WRITE_TOOLS`: it temporarily
creates one `Interp` numerical feature under the runtime lock and visible-main
identity guard. It is not node-free read-only evaluation. It never solves,
saves, creates datasets, or changes physics, geometry, mesh, study, parameters,
or existing result nodes. Do not invoke while another client edits the model.

Coordinates are Nx3 global Cartesian points; supported units are exactly `m`,
`mm`, `um`, and `nm`. Inputs are converted to meters and transposed to 3xN.
Existing geometry must be 3D with length unit `m`; other geometry units fail
closed without changing the geometry. Expressions must be fully qualified if
their scope would otherwise be ambiguous. Evaluation is restricted to the
explicit requested domain IDs, with extrapolation disabled.

Only a Solution dataset with matching solution/component/geometry is accepted.
The stored solution must be initialized, nonempty, and have valid SolutionInfo.
Frequency is read for the exact outer/inner tuple using getPNames/getPvals/getUnits;
`freq` must have unit `Hz` and match within relative tolerance 1e-9 or absolute
tolerance 1e-6 Hz. Multiple outer solutions require explicit selection.
No parameter overrides or solver-setting changes are used.

Results preserve expression and point order. Each expression has `values`, one
per input point, with `real`, `imag`, `magnitude`, and `phase_rad`. Coordinates
are returned as `coordinates_m`; frequency and selected solution indices are
included. Shapes must be [expression][1][point]; nonfinite values are rejected.

A UUID tag is checked against the numerical inventory before creation. Only
the successfully created owned tag is removed in finally, including on setter,
evaluation, shape, and serialization failures. Inventory equality is verified.
Cleanup errors are explicit failures, never successful restoration claims.
Normal completion restores the structural numerical tree, but evaluation caches,
dirty flags, and other internal state need not be byte-identical. Process death
or connection loss can prevent cleanup. No automatic solver or evaluation retry
is performed.

The implementation uses documented COMSOL 6.4 Interp and SolutionInfo APIs:
https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_results.52.082.html
https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/api/com/comsol/model/SolutionInfo.html

Validation is covered by offline fake-Java tests; first live use still needs
explicit authorization. Restart/reconnect the MCP service to expose the newly
registered tool; no full Codex restart is inherently required.
