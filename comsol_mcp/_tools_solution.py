"""Solution inspection and guarded temporary point interpolation."""
from __future__ import annotations

from comsol_mcp._model import _require_visible_main
from comsol_mcp._model_ops import _inspect_solution, _read_solution_dof_field
from comsol_mcp._state import _run_tool
from comsol_mcp._point_ops import _evaluate_solution_points, _PointDiagnostics


def evaluate_solution_points(
    expressions: list[str], coordinates: list[list[float]], dataset: str,
    solution_tag: str, component: str, geometry: str, domain_ids: list[int],
    solnum: int, expected_frequency_hz: float, coordinate_unit: str = "m",
    outersolnum: int | None = None,
) -> str:
    """Evaluate stored fields using one guarded temporary Interp numerical node.

    Uses an existing Solution dataset; never solves or saves. Inputs are Nx3
    coordinates in m/mm/um/nm; existing geometry must use meters. Cleanup restores
    the structural numerical tree on success, not byte-identical internal state.
    This is SAFE_VISIBLE_MAIN_WRITE, not node-free read-only evaluation.
    """
    diagnostic = _PointDiagnostics()

    def _impl():
        with diagnostic.step("current_model_identity"):
            model = _require_visible_main("evaluate_solution_points")
        return _evaluate_solution_points(
            model, expressions, coordinates, dataset, solution_tag, component,
            geometry, domain_ids, solnum, expected_frequency_hz, coordinate_unit,
            outersolnum, _diagnostic=diagnostic,
        )
    return _run_tool("evaluate_solution_points", lambda: diagnostic.run(_impl))


def inspect_solution(sol_tag: str) -> str:
    """Read an existing Solution using documented Java getters only.

    Returns initialization, complete parameter values, bounded real/imaginary
    vector summaries and basic SolutionInfo metadata. Unavailable getters are
    reported independently, without retries. No nodes are created or changed.
    Vector getters read the full default solution into memory, but return only
    its length, first eight entries and an all-finite flag (empty: vacuously true).
    This does not evaluate p or provide mesh/DOF mappings or physical units.
    """
    def _impl():
        tag = sol_tag.strip()
        if not tag:
            raise ValueError("sol_tag is required.")
        model = _require_visible_main("inspect_solution")
        return _inspect_solution(model, tag)

    # Serialize access with solve/write tools; classification remains SAFE_READ.
    return _run_tool("inspect_solution", _impl)


def read_solution_dof_field(
    sol_tag: str,
    field_name: str,
    solnum: int | None = None,
    offset: int = 0,
    limit: int = 100,
) -> str:
    """Read exact-name DOF point values from an existing solution, without interpolation.

    Returns available field names, validated mappings and a bounded page (limit
    1..10000). Coordinates use model length units; g_coords, when available,
    use geometry length units. Missing dimensions and unavailable values are
    null, with getter availability reported separately. p_real/p_imag denote
    the requested field's raw DOF values, not an evaluated pressure expression.
    Only documented Java getters are used, once each, without fallback/retry.
    """
    def _impl():
        if not sol_tag.strip():
            raise ValueError("sol_tag is required.")
        if not isinstance(field_name, str) or not field_name:
            raise ValueError("field_name is required (exact DOF name).")
        for name, value, minimum in (("offset", offset, 0), ("limit", limit, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if limit > 10000:
            raise ValueError("limit must be <= 10000.")
        if solnum is not None and (type(solnum) is not int or solnum < 1):
            raise ValueError("solnum must be a positive integer or null.")
        model = _require_visible_main("read_solution_dof_field")
        return _read_solution_dof_field(model, sol_tag.strip(), field_name, solnum, offset, limit)

    return _run_tool("read_solution_dof_field", _impl)


def register(mcp_instance) -> None:
    mcp_instance.add_tool(evaluate_solution_points)
    mcp_instance.add_tool(inspect_solution)
    mcp_instance.add_tool(read_solution_dof_field)
