from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo


UK_TIMEZONE = ZoneInfo(
    "Europe/London"
)

STATE_ALIGNED = "ALIGNED"
ADVANCEMENT_REQUIRED = "ADVANCEMENT REQUIRED"
MULTIPLE_ADVANCEMENTS_REQUIRED = (
    "MULTIPLE ADVANCEMENTS REQUIRED"
)
STATE_AHEAD = "STATE AHEAD"
SEASON_NOT_STARTED = "SEASON NOT STARTED"
SEASON_COMPLETE = "SEASON COMPLETE"

WINDOW_EARLY = "EARLY"
WINDOW_ON_TIME = "ON TIME"
WINDOW_LATE = "AFTER DEADLINE"
WINDOW_UNAVAILABLE = "UNAVAILABLE"

SECONDS_PER_HOUR = 60 * 60

EARLY_ANALYSIS_HOURS = 72
URGENT_DEADLINE_HOURS = 24


@dataclass(frozen=True)
class GameweekDeadline:
    """Represent one official FPL Gameweek event."""

    gameweek: int
    name: str

    deadline_utc: datetime
    deadline_local: datetime

    deadline_passed: bool

    is_previous: bool
    is_current: bool
    is_next: bool

    finished: bool
    data_checked: bool


@dataclass(frozen=True)
class DeadlineIntelligence:
    """Represent Airaola's official season-clock assessment."""

    checked_at_utc: datetime
    checked_at_local: datetime

    saved_gameweek: int

    official_previous_gameweek: int | None
    official_current_gameweek: int | None
    official_next_gameweek: int | None

    active_gameweek: int | None
    next_deadline_gameweek: int | None

    next_deadline_utc: datetime | None
    next_deadline_local: datetime | None

    hours_until_deadline: float | None

    deadline_window: str
    state_status: str

    advancement_count: int
    advancement_target: int | None
    safe_to_advance: bool

    season_started: bool
    season_complete: bool

    recommendation: str

    events: tuple[GameweekDeadline, ...]


def _parse_datetime(
    value: Any,
) -> datetime:
    """Parse an FPL ISO datetime into timezone-aware UTC."""

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            "FPL deadline time must be an ISO datetime string."
        )

    normalised_value = value.strip()

    if not normalised_value:
        raise ValueError(
            "FPL deadline time cannot be empty."
        )

    if normalised_value.endswith("Z"):
        normalised_value = (
            normalised_value[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            normalised_value
        )
    except ValueError as error:
        raise ValueError(
            "FPL deadline time is not a valid "
            f"ISO datetime: {value}"
        ) from error

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def _normalise_now(
    now: datetime | None,
) -> datetime:
    """Return a timezone-aware UTC analysis timestamp."""

    if now is None:
        return datetime.now(
            timezone.utc
        )

    if not isinstance(
        now,
        datetime,
    ):
        raise ValueError(
            "Deadline analysis time must be a datetime."
        )

    if now.tzinfo is None:
        return now.replace(
            tzinfo=timezone.utc
        )

    return now.astimezone(
        timezone.utc
    )


def _validate_saved_gameweek(
    saved_gameweek: int,
) -> int:
    """Validate the Gameweek stored in manager state."""

    try:
        parsed_gameweek = int(
            saved_gameweek
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Saved Gameweek must be an integer."
        ) from error

    if parsed_gameweek < 1:
        raise ValueError(
            "Saved Gameweek must be at least 1."
        )

    return parsed_gameweek


def _extract_events(
    bootstrap_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract and validate the official FPL event list."""

    if not isinstance(
        bootstrap_data,
        dict,
    ):
        raise ValueError(
            "Bootstrap data must be a dictionary."
        )

    raw_events = bootstrap_data.get(
        "events"
    )

    if not isinstance(
        raw_events,
        list,
    ):
        raise ValueError(
            "FPL bootstrap data does not contain "
            "an events list."
        )

    if not raw_events:
        raise ValueError(
            "FPL bootstrap events list is empty."
        )

    required_columns = {
        "id",
        "name",
        "deadline_time",
        "is_previous",
        "is_current",
        "is_next",
        "finished",
        "data_checked",
    }

    events: list[dict[str, Any]] = []

    for raw_event in raw_events:
        if not isinstance(
            raw_event,
            dict,
        ):
            raise ValueError(
                "Every FPL event must be an object."
            )

        missing_columns = (
            required_columns.difference(
                raw_event
            )
        )

        if missing_columns:
            raise ValueError(
                "FPL event is missing deadline fields: "
                + ", ".join(
                    sorted(
                        missing_columns
                    )
                )
            )

        gameweek = int(
            raw_event["id"]
        )

        if gameweek < 1:
            raise ValueError(
                "FPL event IDs must be positive."
            )

        events.append(
            raw_event
        )

    events.sort(
        key=lambda event: int(
            event["id"]
        )
    )

    gameweek_ids = [
        int(
            event["id"]
        )
        for event in events
    ]

    if len(
        gameweek_ids
    ) != len(
        set(
            gameweek_ids
        )
    ):
        raise ValueError(
            "FPL events contain duplicate Gameweek IDs."
        )

    return events


def _single_flagged_gameweek(
    events: list[dict[str, Any]],
    flag: str,
) -> int | None:
    """Return the sole Gameweek carrying one FPL event flag."""

    flagged = [
        int(
            event["id"]
        )
        for event in events
        if bool(
            event.get(
                flag,
                False,
            )
        )
    ]

    if len(
        flagged
    ) > 1:
        raise ValueError(
            f"Multiple FPL events are marked {flag}."
        )

    if not flagged:
        return None

    return flagged[0]


def _build_event_records(
    events: list[dict[str, Any]],
    checked_at_utc: datetime,
) -> tuple[GameweekDeadline, ...]:
    """Convert raw FPL events into immutable deadline records."""

    records: list[GameweekDeadline] = []

    for event in events:
        deadline_utc = _parse_datetime(
            event["deadline_time"]
        )

        records.append(
            GameweekDeadline(
                gameweek=int(
                    event["id"]
                ),
                name=str(
                    event["name"]
                ),
                deadline_utc=deadline_utc,
                deadline_local=(
                    deadline_utc.astimezone(
                        UK_TIMEZONE
                    )
                ),
                deadline_passed=(
                    checked_at_utc
                    >= deadline_utc
                ),
                is_previous=bool(
                    event["is_previous"]
                ),
                is_current=bool(
                    event["is_current"]
                ),
                is_next=bool(
                    event["is_next"]
                ),
                finished=bool(
                    event["finished"]
                ),
                data_checked=bool(
                    event["data_checked"]
                ),
            )
        )

    return tuple(
        records
    )


def _resolve_next_deadline(
    event_records: tuple[
        GameweekDeadline,
        ...
    ],
    official_next_gameweek: int | None,
) -> GameweekDeadline | None:
    """Resolve the next upcoming official deadline."""

    if official_next_gameweek is not None:
        for event in event_records:
            if (
                event.gameweek
                == official_next_gameweek
            ):
                return event

    future_events = [
        event
        for event in event_records
        if not event.deadline_passed
    ]

    if not future_events:
        return None

    return min(
        future_events,
        key=lambda event: (
            event.deadline_utc
        ),
    )


def _resolve_active_gameweek(
    official_previous_gameweek: int | None,
    official_current_gameweek: int | None,
    official_next_gameweek: int | None,
) -> int | None:
    """
    Resolve the Gameweek Airaola should currently be managing.

    Before a deadline, FPL commonly marks the approaching event
    as is_next. After a deadline, it may become is_current.
    """

    if official_current_gameweek is not None:
        return official_current_gameweek

    if official_next_gameweek is not None:
        return official_next_gameweek

    if official_previous_gameweek is not None:
        return official_previous_gameweek

    return None


def _deadline_window(
    hours_until_deadline: float | None,
) -> str:
    """Classify when analysis is occurring relative to deadline."""

    if hours_until_deadline is None:
        return WINDOW_UNAVAILABLE

    if hours_until_deadline < 0:
        return WINDOW_LATE

    if hours_until_deadline > EARLY_ANALYSIS_HOURS:
        return WINDOW_EARLY

    return WINDOW_ON_TIME


def _build_recommendation(
    state_status: str,
    saved_gameweek: int,
    advancement_target: int | None,
    next_deadline_gameweek: int | None,
    hours_until_deadline: float | None,
) -> str:
    """Create a human-readable lifecycle recommendation."""

    if state_status == STATE_AHEAD:
        return (
            "The saved manager state is ahead of the "
            "official FPL season clock. Do not advance or "
            "process another Gameweek until official data "
            "catches up."
        )

    if state_status == MULTIPLE_ADVANCEMENTS_REQUIRED:
        return (
            f"The saved state is multiple Gameweeks behind. "
            f"Advance cautiously from Gameweek "
            f"{saved_gameweek} toward Gameweek "
            f"{advancement_target}, one lifecycle step at "
            "a time."
        )

    if state_status == ADVANCEMENT_REQUIRED:
        return (
            f"Advance the saved state from Gameweek "
            f"{saved_gameweek} to Gameweek "
            f"{advancement_target} before running the next "
            "weekly management cycle."
        )

    if state_status == SEASON_NOT_STARTED:
        return (
            "The official FPL season has not started. "
            "Initial squad planning may continue, but no "
            "Gameweek advancement is required."
        )

    if state_status == SEASON_COMPLETE:
        return (
            "The official FPL season is complete. "
            "Do not advance the saved Gameweek further."
        )

    if (
        hours_until_deadline is not None
        and hours_until_deadline <= URGENT_DEADLINE_HOURS
        and hours_until_deadline >= 0
    ):
        return (
            f"State is aligned. Gameweek "
            f"{next_deadline_gameweek} closes in "
            f"{hours_until_deadline:.1f} hours, so complete "
            "the weekly analysis soon."
        )

    if (
        next_deadline_gameweek is not None
        and hours_until_deadline is not None
    ):
        return (
            f"State is aligned. The Gameweek "
            f"{next_deadline_gameweek} deadline is "
            f"{hours_until_deadline:.1f} hours away."
        )

    return (
        "State is aligned with the official FPL season clock."
    )


def analyse_deadline_intelligence(
    bootstrap_data: dict[str, Any],
    saved_gameweek: int,
    now: datetime | None = None,
) -> DeadlineIntelligence:
    """
    Compare persistent manager state with official FPL events.

    This function only analyses and recommends. It never mutates
    manager state or advances the saved Gameweek.
    """

    checked_at_utc = _normalise_now(
        now
    )

    checked_at_local = (
        checked_at_utc.astimezone(
            UK_TIMEZONE
        )
    )

    parsed_saved_gameweek = (
        _validate_saved_gameweek(
            saved_gameweek
        )
    )

    events = _extract_events(
        bootstrap_data
    )

    official_previous_gameweek = (
        _single_flagged_gameweek(
            events=events,
            flag="is_previous",
        )
    )

    official_current_gameweek = (
        _single_flagged_gameweek(
            events=events,
            flag="is_current",
        )
    )

    official_next_gameweek = (
        _single_flagged_gameweek(
            events=events,
            flag="is_next",
        )
    )

    event_records = _build_event_records(
        events=events,
        checked_at_utc=checked_at_utc,
    )

    next_deadline_event = (
        _resolve_next_deadline(
            event_records=event_records,
            official_next_gameweek=(
                official_next_gameweek
            ),
        )
    )

    active_gameweek = (
        _resolve_active_gameweek(
            official_previous_gameweek=(
                official_previous_gameweek
            ),
            official_current_gameweek=(
                official_current_gameweek
            ),
            official_next_gameweek=(
                official_next_gameweek
            ),
        )
    )

    season_started = any(
        event.deadline_passed
        for event in event_records
    )

    season_complete = all(
        event.finished
        for event in event_records
    )

    next_deadline_utc = (
        next_deadline_event.deadline_utc
        if next_deadline_event is not None
        else None
    )

    next_deadline_local = (
        next_deadline_event.deadline_local
        if next_deadline_event is not None
        else None
    )

    next_deadline_gameweek = (
        next_deadline_event.gameweek
        if next_deadline_event is not None
        else None
    )

    hours_until_deadline: float | None = None

    if next_deadline_utc is not None:
        seconds_until_deadline = (
            next_deadline_utc
            - checked_at_utc
        ).total_seconds()

        hours_until_deadline = round(
            seconds_until_deadline
            / SECONDS_PER_HOUR,
            2,
        )

    advancement_count = 0
    advancement_target: int | None = None
    safe_to_advance = False

    if season_complete:
        state_status = SEASON_COMPLETE
    elif (
        not season_started
        and official_current_gameweek is None
        and official_previous_gameweek is None
    ):
        state_status = SEASON_NOT_STARTED
    elif active_gameweek is None:
        state_status = STATE_ALIGNED
    elif parsed_saved_gameweek > active_gameweek:
        state_status = STATE_AHEAD
    elif parsed_saved_gameweek == active_gameweek:
        state_status = STATE_ALIGNED
    else:
        advancement_count = (
            active_gameweek
            - parsed_saved_gameweek
        )

        advancement_target = (
            active_gameweek
        )

        safe_to_advance = (
            advancement_count == 1
        )

        if advancement_count == 1:
            state_status = (
                ADVANCEMENT_REQUIRED
            )
        else:
            state_status = (
                MULTIPLE_ADVANCEMENTS_REQUIRED
            )

    deadline_window = _deadline_window(
        hours_until_deadline
    )

    recommendation = _build_recommendation(
        state_status=state_status,
        saved_gameweek=parsed_saved_gameweek,
        advancement_target=(
            advancement_target
        ),
        next_deadline_gameweek=(
            next_deadline_gameweek
        ),
        hours_until_deadline=(
            hours_until_deadline
        ),
    )

    return DeadlineIntelligence(
        checked_at_utc=checked_at_utc,
        checked_at_local=checked_at_local,
        saved_gameweek=parsed_saved_gameweek,
        official_previous_gameweek=(
            official_previous_gameweek
        ),
        official_current_gameweek=(
            official_current_gameweek
        ),
        official_next_gameweek=(
            official_next_gameweek
        ),
        active_gameweek=active_gameweek,
        next_deadline_gameweek=(
            next_deadline_gameweek
        ),
        next_deadline_utc=next_deadline_utc,
        next_deadline_local=(
            next_deadline_local
        ),
        hours_until_deadline=(
            hours_until_deadline
        ),
        deadline_window=deadline_window,
        state_status=state_status,
        advancement_count=(
            advancement_count
        ),
        advancement_target=(
            advancement_target
        ),
        safe_to_advance=safe_to_advance,
        season_started=season_started,
        season_complete=season_complete,
        recommendation=recommendation,
        events=event_records,
    )