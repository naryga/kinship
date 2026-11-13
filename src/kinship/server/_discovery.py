"""Auto-registration of tool, resource and prompt modules.

Each public module in `server/tools/`, `server/resources/` and
`server/prompts/` exposes `register(mcp)`. Modules whose name starts with `_`
are helpers and are skipped. Adding a module is enough; there is no shared list
to edit.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from mcp.server.mcpserver import MCPServer


def register_package(mcp: MCPServer, package: ModuleType) -> list[str]:
    """Import every public module in `package` and call its `register(mcp)`.

    Returns the registered module names, in the (alphabetical) order used.
    """
    registered: list[str] = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda info: info.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        register = getattr(module, "register", None)
        if not callable(register):
            raise TypeError(f"{module.__name__} must define register(mcp)")
        register(mcp)
        registered.append(info.name)
    return registered
