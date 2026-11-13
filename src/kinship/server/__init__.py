"""The MCP server: auth, per-request owner context, tools, resources and prompts.

Nothing in this package queries the database itself. It goes through
`kinship.repo` (see `kinship.server.context.repo_session`).
"""
