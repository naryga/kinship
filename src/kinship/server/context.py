"""Per-request owner context: how tools find out who is asking.

Tools, resources and prompts read data like this, and in no other way:

    from kinship.server.context import repo_session

    with repo_session() as repo:
        counts = repo.counts()

`repo_session()` yields a `kinship.repo.Repository` bound to the authenticated
owner on a fresh read-only transaction. Outside an authenticated MCP request
it raises `NoOwnerError`.

How the owner gets here: `BearerAuthMiddleware` puts the owner id in the ASGI
scope of the HTTP request. `OwnerContextMiddleware` (MCP-level middleware, run
by the SDK around every request handler) reads it from the HTTP request that
carried *this* MCP message and sets a context variable for the duration of the
handler. The binding therefore does not depend on which task the SDK runs the
handler in, and synchronous tools (run in a worker thread) inherit it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext

from kinship.repo import Repository, SessionFactory, open_repository
from kinship.server.auth import OWNER_SCOPE_KEY


class NoOwnerError(RuntimeError):
    """Data was requested outside an authenticated request."""


@dataclass(frozen=True)
class RequestOwner:
    owner_id: int
    session_factory: SessionFactory


_current: ContextVar[RequestOwner | None] = ContextVar("kinship_request_owner", default=None)


def _require() -> RequestOwner:
    current = _current.get()
    if current is None:
        raise NoOwnerError("No authenticated owner in this context")
    return current


def current_owner_id() -> int:
    """The authenticated owner's id; raises `NoOwnerError` if there is none."""
    return _require().owner_id


@contextmanager
def repo_session() -> Iterator[Repository]:
    """Yield a read-only `Repository` for the authenticated owner."""
    current = _require()
    with open_repository(current.session_factory, current.owner_id) as repo:
        yield repo


@contextmanager
def bind_owner(owner_id: int, session_factory: SessionFactory) -> Iterator[None]:
    """Make `owner_id` the current owner inside the block (used by the middleware and tests)."""
    token = _current.set(RequestOwner(owner_id, session_factory))
    try:
        yield
    finally:
        _current.reset(token)


class OwnerContextMiddleware:
    """MCP middleware: bind the owner of the carrying HTTP request around each handler."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    async def __call__(
        self, ctx: ServerRequestContext[Any, Any], call_next: CallNext
    ) -> HandlerResult:
        scope = getattr(ctx.request, "scope", None)
        owner_id = scope.get(OWNER_SCOPE_KEY) if scope is not None else None
        # Always (re)bind, so a handler never sees an owner inherited from elsewhere.
        token = _current.set(
            RequestOwner(owner_id, self._session_factory) if owner_id is not None else None
        )
        try:
            return await call_next(ctx)
        finally:
            _current.reset(token)
