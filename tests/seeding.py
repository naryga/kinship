"""Small hand-built networks for repository and server tests.

`seed_network` gives an owner the same *shape* of data every time (an import,
companies, connections, messages, a position) with a caller-chosen size and
name tag, so two owners can be told apart and any leak shows up in counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from kinship.db.models import Company, Connection, Import, Message, MyPosition, Owner


@dataclass(frozen=True)
class SeededNetwork:
    owner_id: int
    tag: str
    import_id: int
    company_ids: list[int]
    connection_ids: list[int]

    @property
    def expected_counts(self) -> dict[str, int]:
        """Expected non-zero `Repository.counts()` entries for this owner."""
        return {
            "imports": 1,
            "companies": len(self.company_ids),
            "connections": len(self.connection_ids),
            "messages": len(self.connection_ids),
            "my_positions": 1,
        }


def seed_network(session: Session, owner: Owner, *, connections: int, tag: str) -> SeededNetwork:
    """Add an import, 2 companies, `connections` connections (one message each) and a position."""
    imp = Import(
        owner_id=owner.id,
        source_sha256=f"sha-{tag}",
        exported_at=datetime(2026, 1, 2, tzinfo=UTC),
        file_counts={"Connections.csv": connections},
    )
    companies = [
        Company(owner_id=owner.id, name_normalized=f"{tag}-co-{i}", display_name=f"{tag} Co {i}")
        for i in range(2)
    ]
    session.add_all([imp, *companies])
    session.flush()

    people = [
        Connection(
            owner_id=owner.id,
            profile_url=f"https://www.linkedin.com/in/{tag}-{i}",
            first_name=f"{tag}{i}",
            last_name="Person",
            company_raw=companies[i % 2].display_name,
            company_id=companies[i % 2].id,
            position="Engineer",
            first_seen_import_id=imp.id,
            last_seen_import_id=imp.id,
        )
        for i in range(connections)
    ]
    session.add_all(people)
    session.flush()

    session.add_all(
        Message(
            owner_id=owner.id,
            conversation_id=f"{tag}-conv-{i}",
            from_name=f"{tag}{i} Person",
            body=f"hello from {tag}",
            connection_id=person.id,
        )
        for i, person in enumerate(people)
    )
    session.add(MyPosition(owner_id=owner.id, company=f"{tag} Co 0", title="Engineer"))
    session.flush()

    return SeededNetwork(
        owner_id=owner.id,
        tag=tag,
        import_id=imp.id,
        company_ids=[c.id for c in companies],
        connection_ids=[p.id for p in people],
    )
