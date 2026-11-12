"""Mechanical guard: nothing in `kinship.server` queries the database itself.

Server code (tools, resources, prompts, app) may only reach data through
`kinship.repo`. It must not import `kinship.db` or `sqlalchemy`, and must not
execute statements on a session. Also checks tool-module discovery.
"""

import ast
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer

import kinship.server
from kinship.server import prompts, resources, tools

SERVER_DIR = Path(kinship.server.__file__).parent
MODULES = sorted(SERVER_DIR.rglob("*.py"))
FORBIDDEN_MODULES = ("kinship.db", "sqlalchemy", "psycopg", "alembic")
FORBIDDEN_ATTRS = {"execute", "scalars", "scalar", "_session", "session"}


def _is_forbidden(module: str) -> bool:
    return any(module == m or module.startswith(f"{m}.") for m in FORBIDDEN_MODULES)


def violations(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [
                f"line {node.lineno}: imports {alias.name}"
                for alias in node.names
                if _is_forbidden(alias.name)
            ]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_forbidden(module):
                found.append(f"line {node.lineno}: imports from {module}")
            elif module == "kinship":
                found += [
                    f"line {node.lineno}: imports kinship.{alias.name}"
                    for alias in node.names
                    if alias.name == "db"
                ]
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            found.append(f"line {node.lineno}: uses .{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"select", "text", "Session", "__import__"}
        ):
            found.append(f"line {node.lineno}: calls {node.func.id}()")
    return found


def test_there_are_modules_to_check() -> None:
    names = {str(p.relative_to(SERVER_DIR)) for p in MODULES}
    assert {"app.py", "context.py", "tools/network_summary.py"} <= names


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(SERVER_DIR)))
def test_server_module_does_not_touch_the_database(path: Path) -> None:
    assert violations(path.read_text()) == []


@pytest.mark.parametrize(
    "source",
    [
        "from kinship.db.models import Connection",
        "from kinship.db import models",
        "from kinship import db",
        "import kinship.db.models",
        "import kinship.db.engine as e",
        "from sqlalchemy import select",
        "from sqlalchemy import text",
        "from sqlalchemy.orm import Session",
        "import sqlalchemy as sa",
        "def f(s):\n    return s.execute('select 1')",
        "def f(repo):\n    return repo._session",
        "def f(session):\n    return session.scalars(1)",
    ],
)
def test_guard_catches_violations(source: str) -> None:
    assert violations(source)


# -- discovery ------------------------------------------------------------------


def test_tool_modules_are_discovered_and_helpers_skipped() -> None:
    mcp = MCPServer("test")
    registered = tools.register_all(mcp)
    assert "network_summary" in registered
    assert not any(name.startswith("_") for name in registered)

    on_disk = {p.stem for p in (SERVER_DIR / "tools").glob("*.py") if not p.name.startswith("_")}
    assert set(registered) == on_disk


def test_resource_and_prompt_packages_discover_modules() -> None:
    mcp = MCPServer("test")
    for package, directory in ((resources, "resources"), (prompts, "prompts")):
        on_disk = {
            p.stem for p in (SERVER_DIR / directory).glob("*.py") if not p.name.startswith("_")
        }
        assert set(package.register_all(mcp)) == on_disk
