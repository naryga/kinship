"""Warmth scoring: how good an introduction path a connection is. Pure, no database.

`score_warmth` turns the `Signals` recorded about one connection into a score
and the human-readable reasons behind it. The score is a sum of independent
parts, so every point can be traced to a reason:

| Signal                                   | Points                                    |
|------------------------------------------|-------------------------------------------|
| They wrote you a recommendation          | 30 (once, however many)                   |
| You wrote them a recommendation          | 25 (once)                                 |
| You both worked at the same company      | 15 per company, at most 2 companies       |
| Messages: recency of the last one        | 15 x 0.5 ** (age in days / 365)           |
| Messages: volume                         | 1 per message, at most 10                 |
| A note you wrote about them              | 10 (once)                                 |
| They endorsed you                        | 3 per endorsement, at most 4 endorsements |
| You endorsed them                        | 2 per endorsement, at most 4 endorsements |
| They sent the connection invitation      | 4                                         |
| You sent the connection invitation       | 3                                         |
| The invitation carried a personal message| +3                                        |

Message age is measured from `as_of` (the date of the owner's latest export),
never from the wall clock, so the same data always gives the same score. When
the owner has no messages at all (the export was requested without them), pass
`use_messages=False`: the message parts are skipped for everybody rather than
counted as silence.

How long two people have been connected is deliberately not part of the score:
a connection with nothing recorded scores exactly 0. It only breaks ties, older
connections first (see `rank_key`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

RECOMMENDATION_RECEIVED = 30.0
RECOMMENDATION_GIVEN = 25.0
SHARED_EMPLOYER = 15.0
SHARED_EMPLOYER_CAP = 2
MESSAGE_RECENCY = 15.0
MESSAGE_HALF_LIFE_DAYS = 365.0
MESSAGE_VOLUME = 1.0
MESSAGE_VOLUME_CAP = 10
NOTE = 10.0
ENDORSEMENT_RECEIVED = 3.0
ENDORSEMENT_GIVEN = 2.0
ENDORSEMENT_CAP = 4
INVITED_BY_THEM = 4.0
INVITED_BY_YOU = 3.0
INVITATION_MESSAGE = 3.0

MAX_NAMED = 3
"""How many skills or companies a single reason names before "and N more"."""


@dataclass(frozen=True)
class Signals:
    """Everything recorded between the owner and one connection.

    "received" means from the connection to the owner, "given" the reverse.
    The recommendation tuples hold one entry per recommendation: its date, or
    None when the export has none. The endorsement tuples hold skill names.
    `invitation` is "incoming" when they invited the owner, "outgoing" when the
    owner invited them. `shared_employers` are companies both have worked at.
    """

    message_count: int = 0
    last_message_at: datetime | None = None
    recommendations_received: tuple[date | None, ...] = ()
    recommendations_given: tuple[date | None, ...] = ()
    endorsements_received: tuple[str | None, ...] = ()
    endorsements_given: tuple[str | None, ...] = ()
    note_count: int = 0
    invitation: Literal["incoming", "outgoing"] | None = None
    invitation_had_message: bool = False
    shared_employers: tuple[str, ...] = ()
    connected_on: date | None = None


@dataclass(frozen=True)
class Warmth:
    """A score (0 = nothing recorded) and its reasons, strongest first."""

    score: float
    reasons: tuple[str, ...]


def _years(dates: tuple[date | None, ...]) -> str:
    years = sorted({d.year for d in dates if d is not None})
    return f" in {_listed([str(year) for year in years])}" if years else ""


def _listed(names: list[str]) -> str:
    """ "A", "A and B", "A, B and C", "A, B, C and 2 more"."""
    shown, extra = names[:MAX_NAMED], len(names) - MAX_NAMED
    if extra > 0:
        return f"{', '.join(shown)} and {extra} more"
    if len(shown) <= 1:
        return "".join(shown)
    return f"{', '.join(shown[:-1])} and {shown[-1]}"


def _skills(skills: tuple[str | None, ...]) -> str:
    names = sorted({skill for skill in skills if skill}, key=str.casefold)
    return f" for {_listed(names)}" if names else ""


def message_recency(last_message_at: datetime | None, as_of: datetime) -> float:
    """1.0 for a message on `as_of`, halving every `MESSAGE_HALF_LIFE_DAYS`; 0.0 if undated."""
    if last_message_at is None:
        return 0.0
    age_days = max((as_of - last_message_at).total_seconds() / 86400, 0.0)
    return 0.5 ** (age_days / MESSAGE_HALF_LIFE_DAYS)


def score_warmth(signals: Signals, as_of: datetime, *, use_messages: bool = True) -> Warmth:
    """Score one connection. See the module docstring for the weights.

    `as_of` is the reference date for message recency. Every part that adds to
    the score adds one reason; reasons are ordered by the points they
    contributed. With no signal the score is 0 and the single reason says so.
    """
    parts: list[tuple[float, str]] = []

    if signals.recommendations_received:
        when = _years(signals.recommendations_received)
        parts.append((RECOMMENDATION_RECEIVED, f"Wrote you a recommendation{when}"))
    if signals.recommendations_given:
        when = _years(signals.recommendations_given)
        parts.append((RECOMMENDATION_GIVEN, f"You wrote them a recommendation{when}"))

    if signals.shared_employers:
        employers = sorted(set(signals.shared_employers), key=str.casefold)
        points = SHARED_EMPLOYER * min(len(employers), SHARED_EMPLOYER_CAP)
        parts.append((points, f"You both worked at {_listed(employers)}"))

    if use_messages and signals.message_count > 0:
        count = signals.message_count
        points = MESSAGE_VOLUME * min(count, MESSAGE_VOLUME_CAP)
        points += MESSAGE_RECENCY * message_recency(signals.last_message_at, as_of)
        reason = f"You exchanged {count} message{'' if count == 1 else 's'}"
        if signals.last_message_at is not None:
            last = signals.last_message_at.date().isoformat()
            reason += f", last on {last}" if count > 1 else f" on {last}"
        parts.append((points, reason))

    if signals.note_count > 0:
        parts.append((NOTE, "You wrote a note about them"))

    if signals.endorsements_received:
        count = len(signals.endorsements_received)
        points = ENDORSEMENT_RECEIVED * min(count, ENDORSEMENT_CAP)
        parts.append((points, f"Endorsed you{_skills(signals.endorsements_received)}"))
    if signals.endorsements_given:
        count = len(signals.endorsements_given)
        points = ENDORSEMENT_GIVEN * min(count, ENDORSEMENT_CAP)
        parts.append((points, f"You endorsed them{_skills(signals.endorsements_given)}"))

    if signals.invitation is not None:
        incoming = signals.invitation == "incoming"
        points = INVITED_BY_THEM if incoming else INVITED_BY_YOU
        reason = "They invited you to connect" if incoming else "You invited them to connect"
        if signals.invitation_had_message:
            points += INVITATION_MESSAGE
            reason += ", with a personal note"
        parts.append((points, reason))

    if not parts:
        since = f"; connected since {signals.connected_on.year}" if signals.connected_on else ""
        return Warmth(0.0, (f"No recorded interaction{since}",))

    parts.sort(key=lambda part: -part[0])  # stable: equal parts keep the order above
    return Warmth(round(sum(points for points, _ in parts), 1), tuple(r for _, r in parts))


def rank_key(warmth: Warmth, connected_on: date | None, name: str, row_id: int) -> tuple:
    """Sort key for ranking: score (highest first), then the older connection, name, id."""
    return (-warmth.score, connected_on or date.max, name.casefold(), row_id)
