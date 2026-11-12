"""v0 authentication: a static bearer token maps to an `owner_id`.

`BearerAuthMiddleware` is plain ASGI middleware. Every HTTP request except the
public paths must carry `Authorization: Bearer <token>` with a token from
`KINSHIP_TOKENS`; anything else gets a 401. With no tokens configured, every
protected request is refused (fail closed).

On success the owner id is stored in the ASGI scope under `OWNER_SCOPE_KEY`.
`kinship.server.context.OwnerContextMiddleware` picks it up from the HTTP
request that carried each MCP message.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Iterable, Mapping

from starlette.types import ASGIApp, Receive, Scope, Send

OWNER_SCOPE_KEY = "kinship.owner_id"
PUBLIC_PATHS = frozenset({"/healthz"})

_UNAUTHORIZED_BODY = json.dumps(
    {"error": "unauthorized", "error_description": "A valid bearer token is required."}
).encode()


class BearerAuthMiddleware:
    """Require a known bearer token on every non-public HTTP request."""

    def __init__(
        self,
        app: ASGIApp,
        tokens: Mapping[str, int],
        public_paths: Iterable[str] = PUBLIC_PATHS,
    ) -> None:
        self.app = app
        self._tokens = [(token.encode(), owner_id) for token, owner_id in tokens.items()]
        self._public_paths = frozenset(public_paths)

    def owner_for(self, authorization: bytes | None) -> int | None:
        """The owner id for an `Authorization` header value, or None if it is not valid."""
        if not authorization:
            return None
        scheme, _, credential = authorization.partition(b" ")
        credential = credential.strip()
        if scheme.lower() != b"bearer" or not credential:
            return None
        owner_id: int | None = None
        # Compare against every token, in constant time each, without stopping early.
        for token, token_owner_id in self._tokens:
            if hmac.compare_digest(credential, token):
                owner_id = token_owner_id
        return owner_id

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            # Kinship serves HTTP only; refuse anything else (e.g. WebSocket upgrades).
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["path"] in self._public_paths:
            await self.app(scope, receive, send)
            return

        authorization = next(
            (value for name, value in scope["headers"] if name == b"authorization"), None
        )
        owner_id = self.owner_for(authorization)
        if owner_id is None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode()),
                        (b"www-authenticate", b'Bearer realm="kinship"'),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})
            return

        # A fresh scope dict, so the owner can never bleed into another request.
        await self.app({**scope, OWNER_SCOPE_KEY: owner_id}, receive, send)
