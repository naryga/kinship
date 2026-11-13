"""`draft_intro_request`: a prompt for writing an introduction request. No data access."""

from __future__ import annotations

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

TEMPLATE = """\
Help me ask {connection} for an introduction or referral for this job: {job}

First, gather the facts from my LinkedIn data with the Kinship tools:
1. Look up {connection} with `get_connection`: their current company and title, \
how long we have been connected, and our history (messages, notes, endorsements, \
recommendations).
2. If the job names a company, call `warm_paths` for that company to see why \
{connection} is (or is not) a good path, and whether someone else is a warmer one. \
If I pasted a job description, `match_job` can show which parts of my background fit.

Then draft the message:
- Short: at most about 120 words, suitable for a LinkedIn message.
- Specific: open with the real context we share (how we know each other, the last \
time we talked, a company we both worked at), then say which role I am interested \
in and why I am a plausible fit, in one or two sentences.
- One clear, small ask: for example whether they would be open to a quick chat, or \
to pointing me to the right person. Do not ask for a referral outright unless the \
relationship clearly supports it.
- Low pressure: include an easy out, such as "no worries at all if this isn't \
something you can help with".
- Honest: use only facts returned by the tools or stated by me. Never invent shared \
history, mutual contacts, projects or compliments. If the tools show little or no \
shared context, say so and write a brief, polite note that does not pretend \
otherwise.

Finish by listing the facts you relied on and anything you could not verify, so I \
can check the message before sending it. Do not send anything yourself.
"""


def register(mcp: MCPServer) -> None:
    @mcp.prompt(
        title="Draft an introduction request",
        description=(
            "Draft a short, specific, low-pressure message asking one of your connections "
            "for an introduction or referral for a job, using only real shared context."
        ),
    )
    def draft_intro_request(
        connection: Annotated[
            str,
            Field(description="The connection to ask: their name or LinkedIn profile URL."),
        ],
        job: Annotated[
            str,
            Field(
                description=(
                    "The job you want an introduction for: a title and company, a link, "
                    "or the pasted job description."
                )
            ),
        ],
    ) -> str:
        return TEMPLATE.format(connection=connection.strip(), job=job.strip())
