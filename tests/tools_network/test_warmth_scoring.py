"""`kinship.warmth.score_warmth`: weights, reasons and ordering. No database."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from kinship import warmth
from kinship.warmth import Signals, Warmth, message_recency, rank_key, score_warmth

AS_OF = datetime(2026, 11, 2, 12, tzinfo=UTC)


def score(signals: Signals, **kwargs) -> Warmth:
    return score_warmth(signals, AS_OF, **kwargs)


def test_no_signals_scores_zero_with_a_reason() -> None:
    assert score(Signals(connected_on=date(2019, 5, 1))) == Warmth(
        0.0, ("No recorded interaction; connected since 2019",)
    )
    assert score(Signals()) == Warmth(0.0, ("No recorded interaction",))


@pytest.mark.parametrize(
    ("signals", "points", "reason"),
    [
        (
            Signals(recommendations_received=(date(2023, 8, 21),)),
            30.0,
            "Wrote you a recommendation in 2023",
        ),
        (
            Signals(recommendations_given=(date(2022, 1, 2), None)),
            25.0,
            "You wrote them a recommendation in 2022",
        ),
        (Signals(recommendations_given=(None,)), 25.0, "You wrote them a recommendation"),
        (
            Signals(shared_employers=("Brindle Analytics",)),
            15.0,
            "You both worked at Brindle Analytics",
        ),
        (Signals(shared_employers=("B", "A", "C")), 30.0, "You both worked at A, B and C"),
        (Signals(note_count=2), 10.0, "You wrote a note about them"),
        (Signals(endorsements_received=("SQL",)), 3.0, "Endorsed you for SQL"),
        (
            Signals(endorsements_received=("SQL", "dbt", "SQL", "Airflow", "Spark", "Go")),
            12.0,
            "Endorsed you for Airflow, dbt, Go and 2 more",
        ),
        (
            Signals(endorsements_given=("SQL", "Python")),
            4.0,
            "You endorsed them for Python and SQL",
        ),
        (Signals(endorsements_given=(None,)), 2.0, "You endorsed them"),
        (Signals(invitation="incoming"), 4.0, "They invited you to connect"),
        (Signals(invitation="outgoing"), 3.0, "You invited them to connect"),
        (
            Signals(invitation="outgoing", invitation_had_message=True),
            6.0,
            "You invited them to connect, with a personal note",
        ),
        (
            Signals(invitation="incoming", invitation_had_message=True),
            7.0,
            "They invited you to connect, with a personal note",
        ),
    ],
)
def test_each_signal_has_its_weight_and_reason(signals, points, reason) -> None:
    assert score(signals) == Warmth(points, (reason,))


def test_messages_score_volume_plus_decayed_recency() -> None:
    today = score(Signals(message_count=6, last_message_at=AS_OF))
    assert today == Warmth(6 + 15.0, ("You exchanged 6 messages, last on 2026-11-02",))

    a_year_ago = score(Signals(message_count=1, last_message_at=AS_OF - timedelta(days=365)))
    assert a_year_ago == Warmth(1 + 7.5, ("You exchanged 1 message on 2025-11-02",))

    two_years = score(Signals(message_count=40, last_message_at=AS_OF - timedelta(days=730)))
    assert two_years.score == 10 + 3.8  # volume capped at 10; 15 / 4 = 3.75, rounded

    undated = score(Signals(message_count=2))
    assert undated == Warmth(2.0, ("You exchanged 2 messages",))


def test_message_recency_decays_from_as_of_and_never_exceeds_one() -> None:
    assert message_recency(AS_OF, AS_OF) == 1.0
    assert message_recency(AS_OF - timedelta(days=365), AS_OF) == pytest.approx(0.5)
    assert message_recency(AS_OF + timedelta(days=30), AS_OF) == 1.0  # after the export
    assert message_recency(None, AS_OF) == 0.0
    older = message_recency(AS_OF - timedelta(days=900), AS_OF)
    newer = message_recency(AS_OF - timedelta(days=100), AS_OF)
    assert 0 < older < newer < 1


def test_messages_are_skipped_when_the_owner_has_none() -> None:
    signals = Signals(
        message_count=5, last_message_at=AS_OF, note_count=1, connected_on=date(2020, 1, 1)
    )
    assert score(signals, use_messages=False) == Warmth(10.0, ("You wrote a note about them",))
    only_messages = Signals(message_count=5, last_message_at=AS_OF, connected_on=date(2020, 1, 1))
    assert score(only_messages, use_messages=False).score == 0.0


def test_reasons_are_ordered_by_weight_and_score_is_their_sum() -> None:
    everything = Signals(
        message_count=3,
        last_message_at=AS_OF,
        recommendations_received=(date(2023, 1, 1),),
        recommendations_given=(date(2024, 1, 1),),
        endorsements_received=("SQL",),
        endorsements_given=("Python",),
        note_count=1,
        invitation="incoming",
        invitation_had_message=True,
        shared_employers=("Acme",),
        connected_on=date(2015, 1, 1),
    )
    result = score(everything)
    assert result.reasons == (
        "Wrote you a recommendation in 2023",  # 30
        "You wrote them a recommendation in 2024",  # 25
        "You exchanged 3 messages, last on 2026-11-02",  # 18
        "You both worked at Acme",  # 15
        "You wrote a note about them",  # 10
        "They invited you to connect, with a personal note",  # 7
        "Endorsed you for SQL",  # 3
        "You endorsed them for Python",  # 2
    )
    assert result.score == 30 + 25 + 18 + 15 + 10 + 7 + 3 + 2


def test_caps() -> None:
    many = Signals(
        recommendations_received=(date(2020, 1, 1), date(2021, 1, 1)),
        endorsements_given=tuple("abcdefgh"),
        shared_employers=("A", "B", "C", "D", "E"),
    )
    result = score(many)
    assert result.score == 30 + warmth.SHARED_EMPLOYER * 2 + warmth.ENDORSEMENT_GIVEN * 4
    assert result.reasons == (
        "Wrote you a recommendation in 2020 and 2021",
        "You both worked at A, B, C and 2 more",
        "You endorsed them for a, b, c and 5 more",
    )


def test_a_recommendation_outranks_everything_weaker_combined_less_one() -> None:
    """The strongest signals dominate: one recommendation beats notes and endorsements."""
    recommended = score(Signals(recommendations_received=(None,)))
    busy = score(Signals(note_count=1, endorsements_given=tuple("abcd"), invitation="outgoing"))
    assert recommended.score > busy.score


def test_rank_key_orders_by_score_then_older_connection_then_name_then_id() -> None:
    warm, cold = Warmth(12.0, ("x",)), Warmth(0.0, ("y",))
    people = [
        (cold, date(2020, 1, 1), "Zed Young", 1),
        (cold, None, "Abe Undated", 2),
        (cold, date(2015, 1, 1), "Old Friend", 3),
        (warm, date(2024, 1, 1), "Warm Recent", 4),
        (cold, date(2020, 1, 1), "amy Young", 6),
        (cold, date(2020, 1, 1), "Amy Young", 5),
    ]
    ordered = sorted(people, key=lambda item: rank_key(*item))
    assert [item[3] for item in ordered] == [4, 3, 5, 6, 1, 2]
