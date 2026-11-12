"""Mechanical guard: repository modules build queries only through `base.py` helpers.

Outside `repo/base.py` and `repo/session.py`, repository code must not call
`sqlalchemy.select()` / `text()` itself or reach for the session, so the owner
filter cannot be skipped.
"""

import ast
from pathlib import Path

import pytest

import kinship.repo

REPO_DIR = Path(kinship.repo.__file__).parent
EXEMPT = {"base.py", "session.py"}
FORBIDDEN_CALLS = {"select", "text", "union", "union_all", "insert", "update", "delete"}
FORBIDDEN_NAMES = {"Session", "sessionmaker", "select", "text"}

MODULES = sorted(p for p in REPO_DIR.rglob("*.py") if p.name not in EXEMPT)


def violations(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sqlalchemy"):
            found += [
                f"line {node.lineno}: imports {alias.name} from {node.module}"
                for alias in node.names
                if alias.name in FORBIDDEN_NAMES
            ]
        elif isinstance(node, ast.Attribute):
            if node.attr in {"_session", "session"}:
                found.append(f"line {node.lineno}: touches .{node.attr}")
            elif (
                node.attr in FORBIDDEN_CALLS
                and isinstance(node.value, ast.Name)
                and node.value.id in {"sa", "sqlalchemy"}
            ):
                found.append(f"line {node.lineno}: uses {node.value.id}.{node.attr}")
    return found


def test_there_are_modules_to_check() -> None:
    assert any(p.name == "imports.py" for p in MODULES)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_repo_module_uses_only_scoped_helpers(path: Path) -> None:
    assert violations(path.read_text()) == []


@pytest.mark.parametrize(
    "source",
    [
        "from sqlalchemy import select",
        "from sqlalchemy.orm import Session",
        "import sqlalchemy as sa\nsa.select(1)",
        "import sqlalchemy as sa\nsa.text('select 1')",
        "class R:\n    def f(self):\n        return self._session.execute(1)",
    ],
)
def test_guard_catches_violations(source: str) -> None:
    assert violations(source)
