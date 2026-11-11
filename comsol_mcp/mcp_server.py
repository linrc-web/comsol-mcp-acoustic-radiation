#!/usr/bin/env python3
"""COMSOL MCP Server - entrypoint and tool registration.

Server-first MCP bridge for COMSOL Multiphysics. It manages a COMSOL
Multiphysics Server process (or connects to an existing one), then drives the
same server-side model that a COMSOL Desktop client can visualize.
"""

import os

# Preload NumPy during MCP startup, before the COMSOL worker is created.
import numpy

import comsol_mcp._server as _srv
from comsol_mcp._server import mcp, _setup_logging
from comsol_mcp._connection import _ensure_client_shell
from comsol_mcp._state import _write_status
from comsol_mcp._dispatch import AsyncToolRegistry
from comsol_mcp._tools_connection import register as _reg_connection
from comsol_mcp._tools_workflow import register as _reg_workflow
from comsol_mcp._tools_model import register as _reg_model
from comsol_mcp._tools_params import register as _reg_params
from comsol_mcp._tools_geometry import register as _reg_geometry
from comsol_mcp._tools_snapshot import register as _reg_snapshot
from comsol_mcp._tools_physics import register as _reg_physics
from comsol_mcp._tools_solver import register as _reg_solver
from comsol_mcp._tools_solution import register as _reg_solution
from comsol_mcp._tools_mesh import register as _reg_mesh

# Register all tools on the shared FastMCP instance at import time.
_registry = AsyncToolRegistry(mcp)
_reg_connection(_registry)
_reg_workflow(_registry)
_reg_model(_registry)
_reg_params(_registry)
_reg_geometry(_registry)
_reg_snapshot(_registry)
_reg_physics(_registry)
_reg_solver(_registry)
_reg_solution(_registry)
_reg_mesh(_registry)


def main() -> None:
    _setup_logging()
    if os.environ.get("COMSOL_MCP_PREINIT_CLIENT") == "1":
        # Reuse the shared client shell; initialization failures stop startup.
        with _srv._runtime_lock:
            _ensure_client_shell("6.4")
    _write_status({"status": "ready"})
    mcp.run()


if __name__ == "__main__":
    main()
