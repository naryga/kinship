"""`kinship://imports/latest`: the owner's newest import and how much data is loaded.

Resource functions run inside the same MCP request handling as tools, so
`repo_session()` resolves the owner of the HTTP request that asked for the
resource: two tokens reading the same URI get their own contents.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from kinship.server.context import repo_session

URI = "kinship://imports/latest"
SHA_PREFIX_LENGTH = 12


def register(mcp: MCPServer) -> None:
    @mcp.resource(
        URI,
        name="imports_latest",
        title="Latest LinkedIn import",
        description=(
            "Your most recent LinkedIn export (by export date): when it was exported and "
            "loaded, how many rows each file had, how many exports are loaded in total, "
            "and current row counts per kind of record."
        ),
        mime_type="application/json",
    )
    def imports_latest() -> str:
        with repo_session() as repo:
            latest = repo.latest_import()
            import_count = len(repo.imports_in_order())
            counts = repo.counts()

        data: dict[str, Any] = {"import_count": import_count, "latest_import": None}
        if latest is not None:
            data["latest_import"] = {
                "exported_at": latest.exported_at.isoformat() if latest.exported_at else None,
                "imported_at": latest.imported_at.isoformat(),
                "sha256_prefix": latest.source_sha256[:SHA_PREFIX_LENGTH],
                "file_counts": latest.file_counts,
            }
        data["row_counts"] = {table: n for table, n in counts.items() if n}
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
