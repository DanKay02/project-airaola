from __future__ import annotations

from datetime import datetime, timezone

import pytest

from airaola.data.deadline_intelligence import (
    ADVANCEMENT_REQUIRED,
    MULTIPLE_ADVANCEMENTS_REQUIRED,
    SEASON_COMPLETE,
    SEASON_NOT_STARTED,
    STATE_AHEAD,
    STATE_ALIGNED,
    WINDOW_EARLY,
    WINDOW_ON_TIME,
    analyse_deadline_intelligence,
)


def _event(
    gameweek: int,
    deadline: str,
    *,
    is_previous: bool = False,
    is_current: bool = False,
    is_next: bool = False,
    finished: bool = False,
    data_checked: bool = False,
) -> dict:
    """Build one synthetic FPL event."""

    return {
        "id": gameweek,
        "name": f"Gameweek {gameweek}",
        "deadline_time": deadline,
        "is_previous": is_previous,
        "is_current": is_current,
        "is_next": is_next,
        "finished": finished,
        "data_checked": data_checked,
    }


def _bootstrap(
    events: list[dict],
) -> dict:
    """Build synthetic bootstrap data."""

    return {
        "events": events,
    }


def test_state_aligned_with_current_gameweek() -> None:
    now = datetime(
        2026,
        8,
        20,
        12,
        0,
        tzinfo=timezone.utc,
    )

    data = _bootstrap(
        [
            _event(
                1,
                "2026-08-14T17:30:00Z",
                is_previous=True,
                finished=True,
                data_checked=True,
            ),
            _event(
                2,
                "2026-08-21T17:30:00Z",
                is_current=True,
            ),
            _event(
                3,
                "2026-08-28T17:30:00Z",
                is_next=True,
            ),
        ]
    )

    result = analyse_deadline_intelligence(
        bootstrap_data=data,
        saved_gameweek=2,
        now=now,
    )

    assert result.state_status == STATE_ALIGNED
    assert result.active_gameweek == 2
    assert result.advancement_count == 0
    assert result.advancement_target is None
    assert result.safe_to_advance is False


def test_one_gameweek_advancement_required() -> None:
    now = datetime(
        2026,
        8,
        20,
        12,
        0,
        tzinfo=timezone.utc,
    )

    data = _bootstrap(
        [
            _event(
                1,
                "2026-08-14T17:30:00Z",
                is_previous=True,
                finished=True,
                data_checked=True,
            ),
            _event(
                2,
                "2026-08-21T17:30:00Z",
                is_current=True,
            ),
            _event(
                3,
                "2026-08-28T17:30:00Z",
                is_next=True,
            ),
        ]
    )

    result = analyse_deadline_intelligence(
        bootstrap_data=data,
        saved_gameweek=1,
        now=now,
    )

    assert result.state_status == ADVANCEMENT_REQUIRED
    assert result.advancement_count == 1
    assert result.advancement_target == 2
    assert result.safe_to_advance is True


def test_multiple_gameweeks_behind() -> None:
    now = datetime(
        2026,
        9,
        3,
        12,
        0,
        tzinfo=timezone.utc,
    )

    data = _bootstrap(
        [
            _event(
                1,
                "2026-08-14T17:30:00Z",
                finished=True,
                data_checked=True,
            ),
            _event(
                2,
                "2026-08-21T17:30:00Z",
                is_previous=True,
                finished=True,
                data_checked=True,
            ),
            _event(
                3,
                "2026-08-28T17:30:00Z",
                is_current=True,
            ),
            _event(
                4,
                "2026-09-04T17:30:00Z",
                is_next=True,
            ),
        ]
    )

    result = analyse_deadline_intelligence(
        bootstrap_data=data,
        saved_gameweek=1,
        now=now,
    )

    assert (
        result.state_status
        == MULTIPLE_ADVANCEMENTS_REQUIRED
    )
    assert result.advancement_count == 2
    assert result.advancement_target == 3
    assert result.safe_to_advance is False


def test_saved_state_ahead_of_official_clock() -> None:
    now = datetime(
        2026,
        8,
        20,
        12,
        0,
        tzinfo=timezone.utc,
    )

    data = _bootstrap(
        [
            _event(
                1,
                "2026-08-14T17:30:00Z",
                is_previous=True,
                finished=True,
                data_checked=True,
            ),
            _event(
                2,
                "2026-08-21T17:30:00Z",
                is_current=True,
            ),
            _event(
                3,
                "2026-08-28T17:30:00Z",
                is_next=True,
            ),
        ]
    )

    result = analyse_deadline_intelligence(
        bootstrap_data=data,
        saved_gameweek=3,
        now=now,
    )

    assert result.state_status == STATE_AHEAD
    assert result.safe_to_advance is False


def test_preseason_is_detected() -> None:
    now = datetime(
        2026,
        8,
        1,
        12,
        0,
        tzinfo=timezone.utc,
    )

    data = _bootstrap(
        [
            _event(
                1,
                "2026-08-14T17:30:00Z",
                is_next=True,
            ),
            _event(
                2,
                "2026-08-21T17:30:00Z",
            ),
        ]
    )

    result = analyse_deadline_intelligence(
        bootstrap_data=data,
        saved_gameweek=1,
        now=now,
    )

    assert result.state_status == SEASON_NOT_STARTED
    assert result.season_started is False
    assert result.season_complete is False


def test_completed_season_is_detected() -> None:
    now = datetime(
        2027,
        5,
        30,
        12,
        0,
        tzinfo=timezone.utc,
    )

    data = _bootstrap(
        [
            _event(
                37,
                "2027-05-16T14:00:00Z",
                finished=True,
                data_checked=True,
            ),
            _event(
                38,
                "2027-05-23T15:00:00Z",
                is_previous=True,
                finished=True,
                data_checked=True,
            ),
        ]
    )

    result = analyse_deadline_intelligence(
        bootstrap_data=data,
        saved_gameweek=38,
        now=now,
    )

    assert result.state_status == SEASON_COMPLETE
    assert result.season_complete is True
    assert result.safe_to_advance is False


def test_early_deadline_window() -> None:
    now = datetime(
        2026,
        8,
        1,
        12,
        0,
        tzinfo=timezone.utc,
    )

    data = _bootstrap(
        [
            _event(
                1,
                "2026-08-14T17:30:00Z",
                is_next=True,
            ),
        ]
    )

    result = analyse_deadline_intelligence(
        bootstrap_data=data,
        saved_gameweek=1,
        now=now,
    )

    assert result.deadline_window == WINDOW_EARLY
    assert result.hours_until_deadline is not None
    assert result.hours_until_deadline > 72


def test_on_time_deadline_window() -> None:
    now = datetime(
        2026,
        8,
        13,
        12,
        0,
        tzinfo=timezone.utc,
    )

    data = _bootstrap(
        [
            _event(
                1,
                "2026-08-14T17:30:00Z",
                is_next=True,
            ),
        ]
    )

    result = analyse_deadline_intelligence(
        bootstrap_data=data,
        saved_gameweek=1,
        now=now,
    )

    assert result.deadline_window == WINDOW_ON_TIME
    assert result.hours_until_deadline is not None
    assert 0 <= result.hours_until_deadline <= 72


def test_duplicate_current_flag_is_rejected() -> None:
    now = datetime(
        2026,
        8,
        20,
        12,
        0,
        tzinfo=timezone.utc,
    )

    data = _bootstrap(
        [
            _event(
                1,
                "2026-08-14T17:30:00Z",
                is_current=True,
            ),
            _event(
                2,
                "2026-08-21T17:30:00Z",
                is_current=True,
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Multiple FPL events are marked is_current",
    ):
        analyse_deadline_intelligence(
            bootstrap_data=data,
            saved_gameweek=1,
            now=now,
        )


def test_missing_events_list_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="does not contain an events list",
    ):
        analyse_deadline_intelligence(
            bootstrap_data={},
            saved_gameweek=1,
        )


def test_invalid_saved_gameweek_is_rejected() -> None:
    data = _bootstrap(
        [
            _event(
                1,
                "2026-08-14T17:30:00Z",
                is_next=True,
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Saved Gameweek must be at least 1",
    ):
        analyse_deadline_intelligence(
            bootstrap_data=data,
            saved_gameweek=0,
        )