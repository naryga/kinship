"""The tool result convention: compact JSON plus a short text summary."""

from __future__ import annotations

import json
from typing import Any

from mcp_types import CallToolResult, TextContent, ToolAnnotations

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
"""Annotations for every Kinship tool: they only read the owner's own data."""


def tool_result(summary: str, data: dict[str, Any]) -> CallToolResult:
    """Build a tool result from a one-line `summary` and JSON-serializable `data`.

    `data` is returned as structured content. The text content is the summary
    followed by the same data as compact JSON, for clients that only read text.
    """
    compact = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return CallToolResult(
        content=[TextContent(type="text", text=f"{summary}\n{compact}")],
        structured_content=json.loads(compact),
    )
