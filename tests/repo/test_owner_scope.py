"""The repository layer requires an owner and never returns another owner's rows."""

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, aliased, sessionmaker

from kinship.db.models import Company, Connection, Import, Message, Owner
from kinship.repo import BaseRepository, OwnerScope, Repository, open_repository, owned_models


def _sql(stmt: sa.Select) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


# -- an owner is required -------------------------------------------------------


@pytest.mark.parametrize("bad", [None, "1", 1.0, True, object()])
def test_owner_id_must_be_an_int(session: Session, bad: object) -> None:
    with pytest.raises(TypeError):
        Repository(session, bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OwnerScope(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1])
def test_owner_id_must_be_positive(session: Session, bad: int) -> None:
    with pytest.raises(ValueError):
        Repository(session, bad)


def test_owner_id_is_a_required_argument(session: Session) -> None:
    with pytest.raises(TypeError):
        Repository(session)  # type: ignore[call-arg]


def test_session_must_be_a_session() -> None:
    with pytest.raises(TypeError):
        Repository(object(), 1)  # type: ignore[arg-type]


# -- every owned model gets the owner filter ------------------------------------


def test_owned_models_covers_every_table_with_owner_id() -> None:
    from kinship.db.models import Base

    with_owner = {t.name for t in Base.metadata.tables.values() if "owner_id" in t.c}
    assert {m.__tablename__ for m in owned_models()} == with_owner
    assert "owners" not in with_owner and len(with_owner) >= 10


@pytest.mark.parametrize("model", owned_models(), ids=lambda m: m.__tablename__)
def test_select_filters_every_owned_model_by_owner(model: type) -> None:
    scope = OwnerScope(42)
    table = model.__tablename__

    for stmt in (scope.select(model), scope.select(model, sa.func.count())):
        compiled = stmt.compile(dialect=postgresql.dialect())
        assert f"WHERE {table}.owner_id = " in str(compiled)
        assert 42 in compiled.params.values()


def test_select_and_owned_reject_models_without_owner_id() -> None:
    scope = OwnerScope(1)
    with pytest.raises(TypeError):
        scope.select(Owner)
    with pytest.raises(TypeError):
        scope.owned(Connection, Owner)
    with pytest.raises(TypeError):
        scope.owned()


def test_owned_filters_each_side_of_a_join() -> None:
    scope = OwnerScope(7)
    stmt = (
        scope.select(Connection)
        .join(Company, Connection.company_id == Company.id)
        .where(scope.owned(Company))
    )
    sql = _sql(stmt)
    assert "connections.owner_id = " in sql
    assert "companies.owner_id = " in sql


# -- isolation with real rows ---------------------------------------------------


def test_counts_are_per_owner(two_owners) -> None:
    for network, repo in ((two_owners.a, two_owners.repo_a), (two_owners.b, two_owners.repo_b)):
        counts = repo.counts()
        assert set(counts) == {m.__tablename__ for m in owned_models()}
        assert {k: v for k, v in counts.items() if v} == network.expected_counts


def test_get_returns_none_for_another_owners_row(two_owners) -> None:
    a, b = two_owners.a, two_owners.b

    own = two_owners.repo_a.get(Connection, a.connection_ids[0])
    assert own is not None and own.owner_id == a.owner_id

    assert two_owners.repo_a.get(Connection, b.connection_ids[0]) is None
    assert two_owners.repo_a.get(Company, b.company_ids[0]) is None
    assert two_owners.repo_a.get(Import, b.import_id) is None
    assert two_owners.repo_b.get(Connection, a.connection_ids[0]) is None
    assert two_owners.repo_a.get(Connection, 10**12) is None


def test_select_count_and_exists_only_see_own_rows(two_owners) -> None:
    a, b = two_owners.a, two_owners.b
    repo_a, repo_b = two_owners.repo_a, two_owners.repo_b

    rows = repo_a.scalars(repo_a.select(Connection))
    assert {r.id for r in rows} == set(a.connection_ids)
    assert {r.owner_id for r in rows} == {a.owner_id}

    names = repo_b.scalars(repo_b.select(Connection, Connection.first_name))
    assert len(names) == 5 and all(name.startswith("beta") for name in names)

    assert repo_a.count(Connection) == 3
    assert repo_b.count(Connection) == 5
    assert repo_a.count(Connection, Connection.first_name.like("beta%")) == 0
    assert repo_a.exists(Connection, Connection.first_name == "alpha0")
    assert not repo_a.exists(Connection, Connection.first_name == "beta0")
    assert not repo_a.exists(Connection, Connection.id == b.connection_ids[0])


def test_latest_import_is_per_owner(two_owners, session: Session, make_owner) -> None:
    latest_a = two_owners.repo_a.latest_import()
    latest_b = two_owners.repo_b.latest_import()
    assert latest_a is not None and latest_a.id == two_owners.a.import_id
    assert latest_b is not None and latest_b.id == two_owners.b.import_id
    assert latest_a.source_sha256 == "sha-alpha"
    assert latest_a.as_dict()["exported_at"].startswith("2026-01-02")

    newer = Import(owner_id=two_owners.a.owner_id, source_sha256="sha-alpha-2")
    session.add(newer)
    session.flush()
    assert two_owners.repo_a.latest_import().id == newer.id
    assert two_owners.repo_b.latest_import().id == two_owners.b.import_id

    assert Repository(session, make_owner("Empty").id).latest_import() is None


# -- the backstop: statements that forget the owner filter ------------------------


def test_backstop_filters_statements_that_forgot_the_owner(two_owners, session: Session) -> None:
    """Even a hand-written statement with no owner filter is scoped when run by a repository."""
    a, b = two_owners.a, two_owners.b
    # Cross-link on purpose: one of A's connections points at B's company.
    session.get(Connection, a.connection_ids[0]).company_id = b.company_ids[0]
    session.flush()

    other_company = aliased(Company)
    statements = {
        "bare": sa.select(Connection.id),
        "join": sa.select(Connection.id, Company.id).join(
            Company, Connection.company_id == Company.id
        ),
        "aliased join": sa.select(Connection.id, other_company.id).join(
            other_company, Connection.company_id == other_company.id
        ),
        "subquery": sa.select(Connection.id).where(
            Connection.company_id.in_(sa.select(Company.id))
        ),
        "messages": sa.select(Message.connection_id),
    }
    # Alternate owners so a cached statement cannot carry the previous owner's id.
    for _ in range(2):
        for network, repo in ((a, two_owners.repo_a), (b, two_owners.repo_b)):
            own_connections = set(network.connection_ids)
            own_companies = set(network.company_ids)
            for name, stmt in statements.items():
                rows = repo.rows(stmt)
                assert rows, name
                assert {row[0] for row in rows} <= own_connections, name
                if len(rows[0]) == 2:
                    assert {row[1] for row in rows} <= own_companies, name

    # The cross-linked connection never surfaces B's company to A.
    joined = two_owners.repo_a.rows(statements["join"])
    assert a.connection_ids[0] not in {row[0] for row in joined}
    assert len(two_owners.repo_a.rows(statements["bare"])) == 3


def test_join_helper_returns_only_own_pairs(two_owners) -> None:
    repo = two_owners.repo_b
    stmt = (
        repo.select(Connection, Connection.id, Company.id)
        .join(Company, Connection.company_id == Company.id)
        .where(repo.owned(Company))
    )
    rows = repo.rows(stmt)
    assert len(rows) == 5
    assert {r[0] for r in rows} == set(two_owners.b.connection_ids)
    assert {r[1] for r in rows} == set(two_owners.b.company_ids)


# -- read-only sessions ---------------------------------------------------------


def test_open_repository_is_read_only_and_owner_bound(engine, owner_with_committed_rows) -> None:
    owner_id = owner_with_committed_rows
    factory = sessionmaker(bind=engine)

    with open_repository(factory, owner_id) as repo:
        assert isinstance(repo, BaseRepository) and repo.owner_id == owner_id
        assert repo.count(Company) == 1
        with pytest.raises(DBAPIError, match="read-only transaction"):
            repo._session.execute(
                sa.insert(Company).values(owner_id=owner_id, name_normalized="x", display_name="X")
            )

    with pytest.raises(TypeError), open_repository(factory, None):  # type: ignore[arg-type]
        pass


@pytest.fixture
def owner_with_committed_rows(engine):
    """A committed owner with one company; removed again afterwards."""
    with Session(engine) as s:
        owner = Owner(display_name="Committed")
        s.add(owner)
        s.flush()
        s.add(Company(owner_id=owner.id, name_normalized="acme", display_name="Acme"))
        s.commit()
        owner_id = owner.id
    try:
        yield owner_id
    finally:
        with Session(engine) as s:
            s.execute(sa.delete(Owner).where(Owner.id == owner_id))
            s.commit()
