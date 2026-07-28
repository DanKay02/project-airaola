from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import requests


FPL_FIXTURES_URL = (
    "https://fantasy.premierleague.com/api/fixtures/"
)


def fetch_fixture_data() -> list[dict[str, Any]]:
    """Download the current FPL fixture calendar."""

    response = requests.get(
        FPL_FIXTURES_URL,
        timeout=30,
        headers={
            "User-Agent": "Project-Airaola/0.1.5",
            "Accept": "application/json",
        },
    )

    response.raise_for_status()

    fixtures = response.json()

    if not isinstance(fixtures, list):
        raise ValueError(
            "FPL fixture response was not returned as a list."
        )

    return fixtures


def build_team_lookup(
    bootstrap_data: dict[str, Any],
) -> dict[int, str]:
    """Create a lookup from FPL team ID to club name."""

    teams = bootstrap_data.get("teams", [])

    if not teams:
        raise ValueError(
            "Bootstrap data does not contain team information."
        )

    return {
        int(team["id"]): str(team["name"])
        for team in teams
    }


def build_fixture_dataframe(
    fixture_data: Iterable[dict[str, Any]],
    bootstrap_data: dict[str, Any],
) -> pd.DataFrame:
    """Convert raw fixture records into a readable table."""

    fixtures = pd.DataFrame(fixture_data)

    required_columns = {
        "id",
        "event",
        "kickoff_time",
        "team_h",
        "team_a",
        "team_h_difficulty",
        "team_a_difficulty",
        "finished",
        "started",
    }

    missing_columns = required_columns.difference(
        fixtures.columns
    )

    if missing_columns:
        raise ValueError(
            "Fixture data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    team_lookup = build_team_lookup(bootstrap_data)

    fixtures["home_team"] = fixtures["team_h"].map(
        team_lookup
    )

    fixtures["away_team"] = fixtures["team_a"].map(
        team_lookup
    )

    fixtures["kickoff_time"] = pd.to_datetime(
        fixtures["kickoff_time"],
        errors="coerce",
        utc=True,
    )

    selected_columns = [
        "id",
        "event",
        "kickoff_time",
        "home_team",
        "away_team",
        "team_h",
        "team_a",
        "team_h_difficulty",
        "team_a_difficulty",
        "started",
        "finished",
    ]

    return fixtures[selected_columns].copy()


def build_team_fixture_view(
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert each match into one row per participating team.

    A fixture therefore produces:
    - one row for the home team
    - one row for the away team
    """

    home_view = pd.DataFrame(
        {
            "fixture_id": fixtures["id"],
            "event": fixtures["event"],
            "kickoff_time": fixtures["kickoff_time"],
            "team_id": fixtures["team_h"],
            "team_name": fixtures["home_team"],
            "opponent_id": fixtures["team_a"],
            "opponent_name": fixtures["away_team"],
            "venue": "H",
            "difficulty": fixtures[
                "team_h_difficulty"
            ],
            "started": fixtures["started"],
            "finished": fixtures["finished"],
        }
    )

    away_view = pd.DataFrame(
        {
            "fixture_id": fixtures["id"],
            "event": fixtures["event"],
            "kickoff_time": fixtures["kickoff_time"],
            "team_id": fixtures["team_a"],
            "team_name": fixtures["away_team"],
            "opponent_id": fixtures["team_h"],
            "opponent_name": fixtures["home_team"],
            "venue": "A",
            "difficulty": fixtures[
                "team_a_difficulty"
            ],
            "started": fixtures["started"],
            "finished": fixtures["finished"],
        }
    )

    team_fixtures = pd.concat(
        [home_view, away_view],
        ignore_index=True,
    )

    team_fixtures["event"] = pd.to_numeric(
        team_fixtures["event"],
        errors="coerce",
    ).astype("Int64")

    team_fixtures["difficulty"] = pd.to_numeric(
        team_fixtures["difficulty"],
        errors="coerce",
    )

    return team_fixtures.sort_values(
        by=[
            "event",
            "kickoff_time",
            "team_name",
        ],
        na_position="last",
    ).reset_index(drop=True)


def find_current_gameweek(
    bootstrap_data: dict[str, Any],
) -> int:
    """Find the next relevant Gameweek from bootstrap data."""

    events = bootstrap_data.get("events", [])

    next_event = next(
        (
            event
            for event in events
            if event.get("is_next")
        ),
        None,
    )

    if next_event is not None:
        return int(next_event["id"])

    current_event = next(
        (
            event
            for event in events
            if event.get("is_current")
        ),
        None,
    )

    if current_event is not None:
        return int(current_event["id"])

    unfinished_events = [
        event
        for event in events
        if not event.get("finished", False)
    ]

    if unfinished_events:
        return int(unfinished_events[0]["id"])

    raise ValueError(
        "Unable to determine the next relevant Gameweek."
    )


def filter_fixture_horizon(
    team_fixtures: pd.DataFrame,
    starting_gameweek: int,
    planning_horizon: int,
) -> pd.DataFrame:
    """Keep fixtures inside the manager's planning horizon."""

    if planning_horizon <= 0:
        raise ValueError(
            "Planning horizon must be greater than zero."
        )

    final_gameweek = (
        starting_gameweek
        + planning_horizon
        - 1
    )

    return team_fixtures[
        team_fixtures["event"].between(
            starting_gameweek,
            final_gameweek,
        )
    ].copy()


def classify_team_gameweeks(
    team_fixtures: pd.DataFrame,
    team_names: list[str],
    starting_gameweek: int,
    planning_horizon: int,
) -> pd.DataFrame:
    """
    Classify each club's Gameweek as blank, normal or double.

    Two or more fixtures are classified as a Double Gameweek.
    """

    gameweeks = range(
        starting_gameweek,
        starting_gameweek + planning_horizon,
    )

    rows: list[dict[str, Any]] = []

    for team_name in sorted(team_names):
        for gameweek in gameweeks:
            matches = team_fixtures[
                (
                    team_fixtures["team_name"]
                    == team_name
                )
                & (
                    team_fixtures["event"]
                    == gameweek
                )
            ]

            fixture_count = len(matches)

            if fixture_count == 0:
                classification = "BLANK"
                opponents = "-"
                average_difficulty = None

            elif fixture_count == 1:
                classification = "NORMAL"
                opponents = format_opponents(matches)
                average_difficulty = float(
                    matches["difficulty"].mean()
                )

            else:
                classification = "DOUBLE"
                opponents = format_opponents(matches)
                average_difficulty = float(
                    matches["difficulty"].mean()
                )

            rows.append(
                {
                    "team_name": team_name,
                    "event": gameweek,
                    "fixture_count": fixture_count,
                    "classification": classification,
                    "opponents": opponents,
                    "average_difficulty": (
                        average_difficulty
                    ),
                }
            )

    return pd.DataFrame(rows)


def format_opponents(
    matches: pd.DataFrame,
) -> str:
    """Create a readable opponent description."""

    descriptions = []

    for _, fixture in matches.iterrows():
        descriptions.append(
            f"{fixture['opponent_name']} "
            f"({fixture['venue']})"
        )

    return ", ".join(descriptions)


def run_fixture_pipeline(
    bootstrap_data: dict[str, Any],
    planning_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete fixture intelligence pipeline."""

    print(
        "Fixture Department: "
        "downloading upcoming schedule..."
    )

    fixture_data = fetch_fixture_data()

    fixtures = build_fixture_dataframe(
        fixture_data,
        bootstrap_data,
    )

    team_fixtures = build_team_fixture_view(
        fixtures
    )

    starting_gameweek = find_current_gameweek(
        bootstrap_data
    )

    horizon_fixtures = filter_fixture_horizon(
        team_fixtures,
        starting_gameweek,
        planning_horizon,
    )

    team_names = [
        str(team["name"])
        for team in bootstrap_data["teams"]
    ]

    gameweek_map = classify_team_gameweeks(
        horizon_fixtures,
        team_names,
        starting_gameweek,
        planning_horizon,
    )

    print(
        "Fixture Department: "
        f"Gameweeks {starting_gameweek} to "
        f"{starting_gameweek + planning_horizon - 1} "
        "analysed."
    )

    return horizon_fixtures, gameweek_map