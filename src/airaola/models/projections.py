from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_GAMEWEEK_MINUTES = 90
DEFAULT_HORIZON = 5

FIXTURE_DIFFICULTY_MULTIPLIERS = {
    1: 1.20,
    2: 1.10,
    3: 1.00,
    4: 0.90,
    5: 0.80,
}

HOME_MULTIPLIER = 1.05
AWAY_MULTIPLIER = 0.95


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """Convert a DataFrame column into a safe numeric series."""

    if column not in dataframe.columns:
        return pd.Series(
            default,
            index=dataframe.index,
            dtype=float,
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).fillna(default)


def calculate_availability_probability(
    players: pd.DataFrame,
) -> pd.Series:
    """Estimate each player's probability of being available."""

    status_probabilities = {
        "a": 1.00,
        "d": 0.75,
        "i": 0.05,
        "s": 0.00,
        "u": 0.00,
        "n": 0.00,
    }

    status_probability = (
        players["status"]
        .map(status_probabilities)
        .fillna(0.50)
        .astype(float)
    )

    if "chance_of_playing_next_round" not in players.columns:
        return status_probability

    chance_probability = (
        pd.to_numeric(
            players["chance_of_playing_next_round"],
            errors="coerce",
        )
        / 100
    )

    return chance_probability.fillna(
        status_probability
    ).clip(
        lower=0.0,
        upper=1.0,
    )


def calculate_minutes_reliability(
    players: pd.DataFrame,
) -> pd.Series:
    """Estimate how reliably each player receives meaningful minutes."""

    minutes = numeric_series(
        players,
        "minutes",
    )

    starts = numeric_series(
        players,
        "starts",
    )

    estimated_appearances = np.maximum(
        starts,
        np.ceil(minutes / 90),
    )

    average_minutes = np.where(
        estimated_appearances > 0,
        minutes / estimated_appearances,
        0.0,
    )

    return pd.Series(
        average_minutes / 90,
        index=players.index,
    ).clip(
        lower=0.0,
        upper=1.0,
    )


def calculate_points_per_90(
    players: pd.DataFrame,
) -> pd.Series:
    """Calculate historic FPL points per 90 minutes."""

    points = numeric_series(
        players,
        "total_points",
    )

    minutes = numeric_series(
        players,
        "minutes",
    )

    points_per_90 = np.where(
        minutes >= 90,
        points / minutes * 90,
        0.0,
    )

    return pd.Series(
        points_per_90,
        index=players.index,
    ).clip(
        lower=0.0,
        upper=15.0,
    )


def calculate_fixture_multiplier(
    difficulty: float,
    venue: str,
) -> float:
    """Return the combined difficulty and venue multiplier."""

    if pd.isna(difficulty):
        difficulty_multiplier = 1.0
    else:
        rounded_difficulty = int(round(float(difficulty)))

        difficulty_multiplier = (
            FIXTURE_DIFFICULTY_MULTIPLIERS.get(
                rounded_difficulty,
                1.0,
            )
        )

    if venue == "H":
        venue_multiplier = HOME_MULTIPLIER
    elif venue == "A":
        venue_multiplier = AWAY_MULTIPLIER
    else:
        venue_multiplier = 1.0

    return difficulty_multiplier * venue_multiplier


def calculate_team_fixture_projections(
    team_fixtures: pd.DataFrame,
    base_points_per_gameweek: dict[str, float],
) -> pd.DataFrame:
    """
    Calculate projected points for every team fixture.

    Each fixture is calculated independently, allowing blanks and
    doubles to emerge naturally from the schedule.
    """

    required_columns = {
        "team_name",
        "event",
        "opponent_name",
        "venue",
        "difficulty",
    }

    missing_columns = required_columns.difference(
        team_fixtures.columns
    )

    if missing_columns:
        raise ValueError(
            "Fixture data is missing projection columns: "
            + ", ".join(sorted(missing_columns))
        )

    fixture_projection = team_fixtures.copy()

    fixture_projection["fixture_multiplier"] = (
        fixture_projection.apply(
            lambda row: calculate_fixture_multiplier(
                difficulty=row["difficulty"],
                venue=row["venue"],
            ),
            axis=1,
        )
    )

    fixture_projection["team_base_points"] = (
        fixture_projection["team_name"]
        .map(base_points_per_gameweek)
        .fillna(0.0)
    )

    fixture_projection["team_fixture_projection"] = (
        fixture_projection["team_base_points"]
        * fixture_projection["fixture_multiplier"]
    )

    return fixture_projection


def build_player_projections(
    players: pd.DataFrame,
    team_fixtures: pd.DataFrame,
    planning_horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """
    Produce fixture-adjusted projections for every player.

    Blank Gameweeks contribute zero fixture points.
    Double Gameweeks contribute one projection per fixture.
    """

    if planning_horizon <= 0:
        raise ValueError(
            "Planning horizon must be greater than zero."
        )

    projected = players.copy()

    form = numeric_series(
        projected,
        "form",
    )

    points_per_game = numeric_series(
        projected,
        "points_per_game",
    )

    points_per_90 = calculate_points_per_90(
        projected
    )

    availability = (
        calculate_availability_probability(
            projected
        )
    )

    minutes_reliability = (
        calculate_minutes_reliability(
            projected
        )
    )

    expected_minutes_per_fixture = (
        DEFAULT_GAMEWEEK_MINUTES
        * minutes_reliability
        * availability
    )

    raw_points_per_fixture = (
        form * 0.45
        + points_per_game * 0.30
        + points_per_90 * 0.25
    )

    baseline_points_per_fixture = (
        raw_points_per_fixture
        * minutes_reliability
        * availability
    )

    projected["availability_probability"] = (
        availability.round(3)
    )

    projected["minutes_reliability"] = (
        minutes_reliability.round(3)
    )

    projected["points_per_90"] = (
        points_per_90.round(2)
    )

    projected["baseline_points_per_fixture"] = (
        baseline_points_per_fixture.round(3)
    )

    team_base_points = (
        projected
        .groupby("team_name")[
            "baseline_points_per_fixture"
        ]
        .mean()
        .to_dict()
    )

    fixture_projection = (
        calculate_team_fixture_projections(
            team_fixtures=team_fixtures,
            base_points_per_gameweek=team_base_points,
        )
    )

    team_fixture_summary = (
        fixture_projection
        .groupby("team_name")
        .agg(
            fixture_count=(
                "event",
                "count",
            ),
            fixture_multiplier_total=(
                "fixture_multiplier",
                "sum",
            ),
            average_fixture_difficulty=(
                "difficulty",
                "mean",
            ),
        )
        .reset_index()
    )

    projected = projected.merge(
        team_fixture_summary,
        on="team_name",
        how="left",
    )

    projected["fixture_count"] = (
        projected["fixture_count"]
        .fillna(0)
        .astype(int)
    )

    projected["fixture_multiplier_total"] = (
        projected["fixture_multiplier_total"]
        .fillna(0.0)
    )

    projected["average_fixture_difficulty"] = (
        projected["average_fixture_difficulty"]
        .round(2)
    )

    projected["expected_minutes"] = (
        expected_minutes_per_fixture
        * projected["fixture_count"]
    ).round(1)

    projected["projected_points"] = (
        projected["baseline_points_per_fixture"]
        * projected["fixture_multiplier_total"]
    ).round(2)

    projected["projected_points_per_fixture"] = np.where(
        projected["fixture_count"] > 0,
        projected["projected_points"]
        / projected["fixture_count"],
        0.0,
    ).round(2)

    projected["projection_value"] = np.where(
        projected["price"] > 0,
        projected["projected_points"]
        / projected["price"],
        0.0,
    ).round(3)

    return projected