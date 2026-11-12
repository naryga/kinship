"""Canonical form for LinkedIn profile URLs.

Every profile-URL column (``URL``, ``Endorser Public Url``, ``Endorsee Public
Url``, ``Connection Profile URL``, ``inviterProfileUrl``, ``inviteeProfileUrl``)
goes through :func:`canonical_profile_url` so rows from different files join.
"""

from __future__ import annotations

import re
from urllib.parse import quote, unquote, urlsplit

_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_SLASHES = re.compile(r"/{2,}")
_LINKEDIN_HOST = "www.linkedin.com"


def canonical_profile_url(raw: str | None) -> str | None:
    """Return ``https://www.linkedin.com/in/<slug>`` style URLs in one form.

    Forces https and the ``www.linkedin.com`` host (for any linkedin.com
    subdomain), drops query, fragment, port and trailing slash, normalizes
    percent-encoding, and lowercases ``/in/<slug>`` paths. Blank input, or text
    with no recognizable host, returns ``None``.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.startswith("//"):
        text = "https:" + text
    elif not _SCHEME.match(text):
        text = "https://" + text
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").rstrip(".")
    except ValueError:
        return None
    if "." not in host or " " in host:
        return None
    if host == "linkedin.com" or host.endswith(".linkedin.com"):
        host = _LINKEDIN_HOST
    path = _SLASHES.sub("/", unquote(parts.path)).strip().rstrip("/")
    if path and not path.startswith("/"):
        path = "/" + path
    if path.lower().startswith("/in/"):
        path = path.lower()
    return f"https://{host}{quote(path, safe='/-._~')}"
