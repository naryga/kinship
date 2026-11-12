"""The seeded synthetic world behind every generated export.

``build_world(seed)`` returns one timeline covering two export dates. Export
version 1 is the world as of ``v1_exported_at``; version 2 is the same world as
of ``v2_exported_at``, after some connections changed jobs, some were removed
and some were added. ``diff_truth(seed)`` lists exactly which profile URLs fall
in each group, so import tests can assert exact counts.

Only the standard library is used, and each section draws from its own
``random.Random`` seeded with a string, so output is stable across platforms.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from functools import lru_cache

from kinship.synth.data import (
    COMPANIES_WITHOUT_CONNECTIONS,
    COMPANY_GROUPS,
    DOMAINS,
    EVENTS,
    FIRST_NAMES,
    LAST_NAMES,
    LOCATIONS,
    OWNER_SKILLS,
    SCHOOLS,
    Domain,
)

V1_EXPORTED_AT = date(2025, 5, 10)
V2_EXPORTED_AT = date(2026, 11, 2)
DEFAULT_CONNECTIONS = 40
MIN_CONNECTIONS = 20
MAX_CONNECTIONS = 1000

_EARLIEST = date(2015, 1, 1)
_PROFILE_BASE = "https://www.linkedin.com/in/"
# The first people hold one spelling each of the multi-spelling companies.
_VARIANT_SLOTS: tuple[tuple[str, str], ...] = tuple(
    (key, raw) for key, raws in COMPANY_GROUPS.items() if len(raws) > 1 for raw in raws
)
_RENAMED_INDEX = next(i for i, (key, _) in enumerate(_VARIANT_SLOTS) if key == "quillfeather")
_RESERVED = max(len(_VARIANT_SLOTS), len(DOMAINS))


@dataclass(frozen=True)
class Connection:
    first_name: str
    last_name: str
    url: str  # canonical profile URL; "" for the few connections without one
    email: str  # "" for most connections
    company: str  # raw spelling as exported; may be ""
    company_key: str  # canonical key into ``World.company_groups``; "" if no company
    position: str
    connected_on: date
    domain: str  # key of the person's topic in ``kinship.synth.data.DOMAINS``

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass(frozen=True)
class Position:
    company: str
    company_key: str
    title: str
    description: str
    location: str
    started_on: date
    finished_on: date | None  # None for the current role
    started_year_only: bool = False
    finished_year_only: bool = False


@dataclass(frozen=True)
class Education:
    school: str
    start: str
    end: str
    notes: str
    degree: str
    activities: str


@dataclass(frozen=True)
class Owner:
    first_name: str
    last_name: str
    url: str
    email: str
    headline: str
    summary: str
    industry: str
    positions: tuple[Position, ...]
    education: tuple[Education, ...]
    skills: tuple[str, ...]

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass(frozen=True)
class Endorsement:
    at: datetime
    skill: str
    first_name: str
    last_name: str
    url: str
    status: str


@dataclass(frozen=True)
class Recommendation:
    first_name: str
    last_name: str
    company: str
    job_title: str
    text: str
    at: datetime
    status: str
    domain: str


@dataclass(frozen=True)
class Invitation:
    from_name: str
    to_name: str
    at: datetime
    message: str
    direction: str  # "OUTGOING" or "INCOMING"
    inviter_url: str
    invitee_url: str


@dataclass(frozen=True)
class Note:
    first_name: str
    last_name: str
    url: str
    text: str
    at: datetime
    edited_at: datetime | None


@dataclass(frozen=True)
class Message:
    conversation_id: str
    title: str
    from_name: str
    sender_url: str
    to_name: str
    recipient_urls: str
    at: datetime
    subject: str
    content: str
    folder: str


@dataclass(frozen=True)
class CompanyFollow:
    organization: str
    at: datetime


@dataclass(frozen=True)
class JobApplication:
    at: datetime
    company: str
    company_key: str  # "" when the owner has no connections there
    job_title: str
    job_url: str


@dataclass(frozen=True)
class SavedJob:
    at: datetime
    job_url: str
    job_title: str
    company: str


@dataclass(frozen=True)
class DiffTruth:
    """Ground truth for the difference between export version 1 and version 2.

    Every tuple holds canonical profile URLs, sorted. Connections without a URL
    never change and are counted in ``blank_url_count`` only.
    """

    v1_exported_at: date
    v2_exported_at: date
    v1_count: int
    v2_count: int
    added: tuple[str, ...]  # in v2 only
    removed: tuple[str, ...]  # in v1 only
    company_changed: tuple[str, ...]  # moved to a different company
    company_renamed: tuple[str, ...]  # same company, different raw spelling
    title_changed: tuple[str, ...]  # Position differs (may overlap company_changed)
    job_changed: tuple[str, ...]  # raw Company or Position differs: union of the three above
    unchanged_count: int  # rows identical in both versions (blank-URL rows included)
    blank_url_count: int


@dataclass(frozen=True)
class World:
    seed: int
    member_id: str
    owner: Owner
    v1_exported_at: date
    v2_exported_at: date
    connections_v1: tuple[Connection, ...]
    connections_v2: tuple[Connection, ...]
    company_groups: dict[str, tuple[str, ...]]
    endorsements_received: tuple[Endorsement, ...]
    endorsements_given: tuple[Endorsement, ...]
    recommendations_received: tuple[Recommendation, ...]
    recommendations_given: tuple[Recommendation, ...]
    invitations: tuple[Invitation, ...]
    notes: tuple[Note, ...]
    messages: tuple[Message, ...]
    company_follows: tuple[CompanyFollow, ...]
    job_applications: tuple[JobApplication, ...]
    saved_jobs: tuple[SavedJob, ...]
    truth: DiffTruth

    def exported_at(self, version: int) -> date:
        _check_version(version)
        return self.v1_exported_at if version == 1 else self.v2_exported_at

    def connections(self, version: int) -> tuple[Connection, ...]:
        """Connections in an export, newest first (the order LinkedIn uses)."""
        _check_version(version)
        return self.connections_v1 if version == 1 else self.connections_v2

    def visible[T](self, events: tuple[T, ...], version: int) -> tuple[T, ...]:
        """Events (anything with an ``at``) that had happened by that export."""
        cutoff = self.exported_at(version)
        return tuple(e for e in events if e.at.date() <= cutoff)  # type: ignore[attr-defined]

    def connections_at(self, company_key: str, version: int) -> tuple[Connection, ...]:
        """Connections whose current company is ``company_key``, any spelling."""
        return tuple(c for c in self.connections(version) if c.company_key == company_key)


def _check_version(version: int) -> None:
    if version not in (1, 2):
        raise ValueError(f"version must be 1 or 2, got {version!r}")


def _rng(seed: int, section: str) -> random.Random:
    return random.Random(f"kinship-synth:{seed}:{section}")


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def _hex(rng: random.Random, digits: int = 8) -> str:
    return f"{rng.getrandbits(digits * 4):0{digits}x}"


def _profile_url(rng: random.Random, first: str, last: str) -> str:
    return f"{_PROFILE_BASE}{_slug(first)}-{_slug(last)}-{_hex(rng)}"


def _between(rng: random.Random, start: date, end: date) -> date:
    if end <= start:
        return start
    return start + timedelta(days=rng.randrange((end - start).days + 1))


def _at(rng: random.Random, day: date) -> datetime:
    return datetime.combine(day, time(rng.randrange(7, 22), rng.randrange(60), rng.randrange(60)))


def _when(rng: random.Random, start: date, end: date) -> datetime:
    return _at(rng, _between(rng, start, end))


def _promote(title: str) -> str:
    for prefix, better in (("Senior ", "Staff "), ("Staff ", "Principal ")):
        if title.startswith(prefix):
            return better + title[len(prefix) :]
    return "Senior " + title


def _domain(key: str) -> Domain:
    return next(d for d in DOMAINS if d.key == key)


def _build_owner(rng: random.Random) -> Owner:
    first, last = "Sam", "Sampleton"
    positions = (
        Position(
            "Gift Health, Inc.",
            "gifthealth",
            "Senior Data Engineer",
            "Own the pharmacy claims data pipelines: ingestion from payers, healthcare "
            "billing reconciliation and the reporting warehouse.\n\n"
            "- Cut rejected-claim turnaround from days to hours\n"
            "- Mentor three engineers and run the incident review rota",
            LOCATIONS[0],
            date(2022, 3, 1),
            None,
        ),
        Position(
            "Acme, Inc.",
            "acme",
            "Data Engineer",
            "Built Airflow batch pipelines and dbt models feeding the analytics warehouse. "
            "Introduced data quality checks that caught schema drift before dashboards broke.",
            LOCATIONS[1],
            date(2018, 6, 1),
            date(2022, 2, 1),
        ),
        Position(
            "Quillfeather Systems",
            "quillfeather",
            "Software Engineer",
            "Backend APIs for the iOS and Android mobile apps, including offline sync and "
            "push notifications. Moved the service from a single server to containers.",
            LOCATIONS[2],
            date(2015, 1, 1),
            date(2018, 5, 1),
            started_year_only=True,
        ),
        Position(
            "Copperline Freight",
            "copperline",
            "Junior Developer",
            "Route-planning tools for freight dispatchers; SQL reports on delivery times "
            "and a small internal web app for tracking shipments.",
            LOCATIONS[3],
            date(2012, 8, 1),
            date(2015, 1, 1),
            finished_year_only=True,
        ),
        Position(
            "Saltmarsh Bank",
            "saltmarsh",
            "Engineering Intern",
            "Wrote scripts for the security audit team to collect access-review evidence "
            "and flag dormant accounts.",
            LOCATIONS[0],
            date(2011, 6, 1),
            date(2011, 9, 1),
        ),
    )
    education = (
        Education(
            SCHOOLS[0],
            "2008",
            "2012",
            "Capstone project on scheduling batch jobs.",
            "Bachelor of Science - BS, Computer Science",
            "Programming club; campus radio",
        ),
        Education(
            SCHOOLS[1],
            "Sep 2016",
            "Jun 2017",
            "",
            "Certificate, Data Engineering",
            "",
        ),
    )
    summary = (
        "Data engineer who likes boring, reliable pipelines. Ten years across freight, "
        "mobile backends and healthcare billing.\n\n"
        "This profile is synthetic test data generated by Kinship."
    )
    return Owner(
        first_name=first,
        last_name=last,
        url=_profile_url(rng, first, last),
        email="sam.sampleton@example.com",
        headline="Senior Data Engineer at Gifthealth | Data pipelines, healthcare billing",
        summary=summary,
        industry="Hospital & Health Care",
        positions=positions,
        education=education,
        skills=OWNER_SKILLS,
    )


def _build_people(
    rng: random.Random, n: int, n_added: int
) -> tuple[list[Connection], list[Connection]]:
    """Return (version 1 people, people added by version 2)."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    while len(pairs) < n + n_added:
        pair = (rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    weighted_keys = [*COMPANY_GROUPS, "acme", "acme", "gifthealth", "gifthealth"]
    spare = list(range(_RESERVED, n))
    blank_company = set(rng.sample(spare, 2))
    blank_url = set(rng.sample([i for i in spare if i not in blank_company], 2))

    people: list[Connection] = []
    for i, (first, last) in enumerate(pairs):
        added = i >= n
        domain = DOMAINS[i % len(DOMAINS)]
        if i < len(_VARIANT_SLOTS):
            key, company = _VARIANT_SLOTS[i]
        elif i == n:  # the first new connection always joins Gifthealth
            key, company = "gifthealth", "Gifthealth"
        else:
            key = rng.choice(weighted_keys)
            company = rng.choice(COMPANY_GROUPS[key])
        position = rng.choice(domain.titles)
        url = _profile_url(rng, first, last)
        email = f"{_slug(first)}.{_slug(last)}@example.com" if rng.random() < 0.2 else ""
        if added:
            connected = _between(
                rng, V1_EXPORTED_AT + timedelta(days=10), V2_EXPORTED_AT - timedelta(days=10)
            )
        else:
            connected = _between(rng, _EARLIEST, V1_EXPORTED_AT - timedelta(days=30))
        if i in blank_company:
            key = company = position = ""
        if i in blank_url:
            url = ""
        people.append(
            Connection(first, last, url, email, company, key, position, connected, domain.key)
        )
    return people[:n], people[n:]


def _newest_first(people: list[Connection]) -> tuple[Connection, ...]:
    return tuple(sorted(people, key=lambda c: (c.connected_on, c.url, c.full_name), reverse=True))


@lru_cache(maxsize=8)
def build_world(seed: int = 42, n_connections: int = DEFAULT_CONNECTIONS) -> World:
    """Build the deterministic world for ``seed``.

    ``n_connections`` is the number of connections in export version 1.
    """
    if not MIN_CONNECTIONS <= n_connections <= MAX_CONNECTIONS:
        raise ValueError(
            f"n_connections must be between {MIN_CONNECTIONS} and {MAX_CONNECTIONS}, "
            f"got {n_connections}"
        )
    n = n_connections
    v1, v2 = V1_EXPORTED_AT, V2_EXPORTED_AT
    owner = _build_owner(_rng(seed, "owner"))
    member_id = f"9{_rng(seed, 'member').randrange(10**8):08d}"

    n_added = max(3, n // 10)
    base, added = _build_people(_rng(seed, "people"), n, n_added)

    # --- what changes between the two exports ---------------------------------
    rng = _rng(seed, "changes")
    n_removed = max(2, n // 13)
    n_company = max(3, n // 8)
    n_title = max(2, n // 13)
    eligible = [i for i in range(len(_VARIANT_SLOTS), n) if base[i].url and base[i].company]
    picked = rng.sample(eligible, n_removed + n_company + n_title)
    removed_idx = set(picked[:n_removed])
    company_idx = picked[n_removed : n_removed + n_company]
    title_idx = picked[n_removed + n_company :]

    later: dict[int, Connection] = {}
    for order, i in enumerate(company_idx):
        person = base[i]
        if order == 0:  # one known move into Acme (or out of it, into Gifthealth)
            key = "gifthealth" if person.company_key == "acme" else "acme"
            company = "Gifthealth" if key == "gifthealth" else "ACME Corp"
        else:
            key = rng.choice([k for k in COMPANY_GROUPS if k != person.company_key])
            company = rng.choice(COMPANY_GROUPS[key])
        # Every other mover also gets a new title.
        position = _promote(person.position) if order % 2 == 0 else person.position
        later[i] = replace(person, company=company, company_key=key, position=position)
    for i in title_idx:
        later[i] = replace(base[i], position=_promote(base[i].position))
    renamed = base[_RENAMED_INDEX]
    other_spelling = next(s for s in COMPANY_GROUPS[renamed.company_key] if s != renamed.company)
    later[_RENAMED_INDEX] = replace(renamed, company=other_spelling)

    after = [later.get(i, p) for i, p in enumerate(base) if i not in removed_idx] + added
    blank_urls = sum(1 for p in base if not p.url)
    job_changed = sorted(base[i].url for i in later)
    truth = DiffTruth(
        v1_exported_at=v1,
        v2_exported_at=v2,
        v1_count=len(base),
        v2_count=len(after),
        added=tuple(sorted(p.url for p in added)),
        removed=tuple(sorted(base[i].url for i in removed_idx)),
        company_changed=tuple(sorted(base[i].url for i in company_idx)),
        company_renamed=(renamed.url,),
        title_changed=tuple(
            sorted(base[i].url for i, p in later.items() if p.position != base[i].position)
        ),
        job_changed=tuple(job_changed),
        unchanged_count=len(base) - len(removed_idx) - len(later),
        blank_url_count=blank_urls,
    )

    linked = [p for p in base if p.url]  # people other files can point at
    me = owner.full_name

    # --- endorsements ---------------------------------------------------------
    rng = _rng(seed, "endorsements")
    received_e, given_e = [], []
    for k in range(max(18, n // 2)):
        p = rng.choice(linked)
        end = v2 if k % 5 == 4 else v1
        start = max(p.connected_on, v1 + timedelta(days=1)) if k % 5 == 4 else p.connected_on
        status = "REJECTED" if k % 9 == 8 else "ACCEPTED"
        received_e.append(
            Endorsement(
                _when(rng, start, end),
                rng.choice(owner.skills),
                p.first_name,
                p.last_name,
                p.url,
                status,
            )
        )
    for k in range(max(12, n // 3)):
        p = rng.choice(linked)
        end = v2 if k % 4 == 3 else v1
        start = max(p.connected_on, v1 + timedelta(days=1)) if k % 4 == 3 else p.connected_on
        given_e.append(
            Endorsement(
                _when(rng, start, end),
                rng.choice(_domain(p.domain).skills),
                p.first_name,
                p.last_name,
                p.url,
                "ACCEPTED",
            )
        )

    # --- recommendations: one distinct topic each -------------------------------
    rng = _rng(seed, "recommendations")
    n_received = 8
    received_r, given_r = [], []
    for k, domain in enumerate(DOMAINS):
        p = base[k]  # person k has topic k and always has a company
        about_owner = k < n_received
        late = k in (n_received - 1, len(DOMAINS) - 1)  # one each only in version 2
        at = _when(rng, v1 + timedelta(days=20), v2) if late else _when(rng, p.connected_on, v1)
        rec = Recommendation(
            p.first_name,
            p.last_name,
            p.company,
            p.position,
            domain.praise.format(name=owner.first_name if about_owner else p.first_name),
            at,
            "HIDDEN" if k == 2 else "VISIBLE",
            domain.key,
        )
        (received_r if about_owner else given_r).append(rec)

    # --- invitations ------------------------------------------------------------
    rng = _rng(seed, "invitations")
    invitations = []
    for k, p in enumerate([*rng.sample(linked, 12), *added]):
        outgoing = k % 2 == 0
        at = _at(rng, p.connected_on - timedelta(days=rng.randrange(6)))
        if k % 3 == 0:
            other = p.first_name if outgoing else owner.first_name
            text = f"Hi {other}, great to meet you at {rng.choice(EVENTS)}. Let's connect."
        else:
            text = ""
        invitations.append(
            Invitation(
                me if outgoing else p.full_name,
                p.full_name if outgoing else me,
                at,
                text,
                "OUTGOING" if outgoing else "INCOMING",
                owner.url if outgoing else p.url,
                p.url if outgoing else owner.url,
            )
        )
    for first, last in (("Pending", "Placeholder"), ("Nobody", "Yetconnected")):
        invitations.append(  # sent, never accepted: no matching connection
            Invitation(
                me,
                f"{first} {last}",
                _when(rng, v1 - timedelta(days=200), v1),
                "",
                "OUTGOING",
                owner.url,
                _profile_url(rng, first, last),
            )
        )

    # --- notes ------------------------------------------------------------------
    rng = _rng(seed, "notes")
    notes = []
    for k, p in enumerate(rng.sample(linked, 6)):
        at = _when(rng, p.connected_on, v1)
        text = f"Met at {rng.choice(EVENTS)}. Knows a lot about {_domain(p.domain).topic}."
        if k == 1:
            text += "\nAsked for an intro to the platform team; follow up next quarter."
        notes.append(
            Note(
                p.first_name,
                p.last_name,
                p.url,
                text,
                at,
                at + timedelta(days=3, hours=1) if k % 3 == 0 else None,
            )
        )

    # --- messages ---------------------------------------------------------------
    rng = _rng(seed, "messages")
    messages = []
    partners = [(p, False) for p in rng.sample(linked, 10)] + [(added[0], True)]
    for p, late in partners:
        domain = _domain(p.domain)
        conversation = f"2-{_hex(rng, 32)}=="
        start = (
            _when(rng, p.connected_on, v2) if late else _when(rng, p.connected_on, v1)
        ) - timedelta(days=1)
        lines = (
            (p, f"Hi {owner.first_name}, {domain.ask}"),
            (None, f"Hi {p.first_name}, good to hear from you. {domain.reply}"),
            (p, "Thanks, that is exactly what I needed.\nTalk soon."),
        )
        for step, (sender, content) in enumerate(lines):
            incoming = sender is not None
            messages.append(
                Message(
                    conversation,
                    "",
                    p.full_name if incoming else me,
                    p.url if incoming else owner.url,
                    me if incoming else p.full_name,
                    owner.url if incoming else p.url,
                    start + timedelta(hours=step * 5, minutes=rng.randrange(50)),
                    f"Question about {domain.topic}" if step == 0 else "",
                    content,
                    "INBOX",
                )
            )

    # --- company follows, job applications, saved jobs ------------------------
    rng = _rng(seed, "jobs")
    with_connections = sorted({p.company_key for p in base if p.company_key} - {"gifthealth"})
    others = [k for k in with_connections if k != "acme"]
    targets = ["acme", *rng.sample(others, min(3, len(others)))]
    follows = [
        CompanyFollow(COMPANY_GROUPS[key][0], _when(rng, _EARLIEST, v1))
        for key in [*targets, "gifthealth"]
    ]
    follows.append(
        CompanyFollow(COMPANIES_WITHOUT_CONNECTIONS[0], _when(rng, v1 + timedelta(days=5), v2))
    )

    job_id = 9000000001
    applications = []
    wanted = [(key, COMPANY_GROUPS[key][-1]) for key in targets]
    wanted += [("", name) for name in COMPANIES_WITHOUT_CONNECTIONS[:2]]
    for k, (key, company) in enumerate(wanted):
        late = k in (3, 5)
        at = _when(rng, v1 + timedelta(days=30), v2) if late else _when(rng, date(2024, 9, 1), v1)
        title = rng.choice(("Staff Data Engineer", "Senior Data Engineer", "Data Platform Lead"))
        applications.append(
            JobApplication(
                at, company, key, title, f"https://www.linkedin.com/jobs/view/{job_id + k}"
            )
        )
    saved = []
    for k in range(5):
        key = rng.choice(with_connections)
        saved.append(
            SavedJob(
                _when(rng, date(2024, 9, 1), v1),
                f"https://www.linkedin.com/jobs/view/{job_id + 100 + k}",
                rng.choice(("Senior Data Engineer", "Analytics Engineering Manager")),
                rng.choice(COMPANY_GROUPS[key]),
            )
        )

    def by_time[T](events: list[T]) -> tuple[T, ...]:
        return tuple(sorted(events, key=lambda e: e.at, reverse=True))  # type: ignore[attr-defined]

    return World(
        seed=seed,
        member_id=member_id,
        owner=owner,
        v1_exported_at=v1,
        v2_exported_at=v2,
        connections_v1=_newest_first(base),
        connections_v2=_newest_first(after),
        company_groups=dict(COMPANY_GROUPS),
        endorsements_received=by_time(received_e),
        endorsements_given=by_time(given_e),
        recommendations_received=tuple(received_r),
        recommendations_given=tuple(given_r),
        invitations=by_time(invitations),
        notes=by_time(notes),
        messages=tuple(sorted(messages, key=lambda m: (m.at, m.conversation_id))),
        company_follows=by_time(follows),
        job_applications=by_time(applications),
        saved_jobs=by_time(saved),
        truth=truth,
    )


def diff_truth(seed: int = 42, n_connections: int = DEFAULT_CONNECTIONS) -> DiffTruth:
    """Which profile URLs change job, are added or are removed between versions."""
    return build_world(seed, n_connections).truth
