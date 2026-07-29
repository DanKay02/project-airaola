from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_HORIZON = 5
MINUTES_PER_MATCH = 90

FIXTURE_DIFFICULTY_MULTIPLIERS = {
    1: 1.20,
    2: 1.10,
    3: 1.00,
    4: 0.90,
    5: 0.80,
}

HOME_MULTIPLIER = 1.05
AWAY_MULTIPLIER = 0.95

GOAL_POINTS = {
    "GKP": 6,
    "DEF": 6,
    "MID": 5,
    "FWD": 4,
}

CLEAN_SHEET_POINTS = {
    "GKP": 4,
    "DEF": 4,
    "MID": 1,
    "FWD": 0,
}


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """Return a safe numeric representation of a column."""

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
    """Estimate the probability that each player is available."""

    status_probabilities = {
        "a": 1.00,
        "d": 0.70,
        "i": 0.05,
        "s": 0.00,
        "u": 0.00,
        "n": 0.00,
    }

    if "status" not in players.columns:
        status_probability = pd.Series(
            1.0,
            index=players.index,
            dtype=float,
        )
    else:
        status_probability = (
            players["status"]
            .map(status_probabilities)
            .fillna(0.50)
            .astype(float)
        )

    if (
        "chance_of_playing_next_round"
        not in players.columns
    ):
        return status_probability

    official_chance = (
        pd.to_numeric(
            players["chance_of_playing_next_round"],
            errors="coerce",
        )
        / 100
    )

    return official_chance.fillna(
        status_probability
    ).clip(
        lower=0.0,
        upper=1.0,
    )


def calculate_start_security(
    players: pd.DataFrame,
) -> pd.Series:
    """Estimate start security relative to positional teammates."""

    starts = numeric_series(
        players,
        "starts",
    )

    comparison = players[
        [
            "team_name",
            "position",
        ]
    ].copy()

    comparison["starts"] = starts

    maximum_starts = (
        comparison
        .groupby(
            [
                "team_name",
                "position",
            ]
        )["starts"]
        .transform("max")
        .replace(0, np.nan)
    )

    security = starts / maximum_starts

    return security.fillna(0.0).clip(
        lower=0.0,
        upper=1.0,
    )


def calculate_minutes_involvement(
    players: pd.DataFrame,
) -> pd.Series:
    """Estimate involvement relative to positional teammates."""

    minutes = numeric_series(
        players,
        "minutes",
    )

    comparison = players[
        [
            "team_name",
            "position",
        ]
    ].copy()

    comparison["minutes"] = minutes

    maximum_minutes = (
        comparison
        .groupby(
            [
                "team_name",
                "position",
            ]
        )["minutes"]
        .transform("max")
        .replace(0, np.nan)
    )

    involvement = minutes / maximum_minutes

    return involvement.fillna(0.0).clip(
        lower=0.0,
        upper=1.0,
    )


def calculate_sample_confidence(
    players: pd.DataFrame,
) -> pd.Series:
    """Reduce confidence in projections based on tiny samples."""

    starts = numeric_series(
        players,
        "starts",
    )

    confidence = np.sqrt(
        starts.clip(lower=0.0) / 15.0
    )

    return pd.Series(
        confidence,
        index=players.index,
    ).clip(
        lower=0.0,
        upper=1.0,
    )


def calculate_minutes_security(
    players: pd.DataFrame,
) -> pd.Series:
    """Estimate future minutes security."""

    start_security = calculate_start_security(
        players
    )

    involvement = calculate_minutes_involvement(
        players
    )

    sample_confidence = calculate_sample_confidence(
        players
    )

    outfield_security = (
        start_security * 0.65
        + involvement * 0.35
    )

    goalkeeper_security = (
        start_security * 0.90
        + involvement * 0.10
    )

    positions = players["position"].astype(str)

    base_security = np.where(
        positions == "GKP",
        goalkeeper_security,
        outfield_security,
    )

    base_security = pd.Series(
        base_security,
        index=players.index,
        dtype=float,
    )

    confidence_multiplier = (
        0.40
        + sample_confidence * 0.60
    )

    return (
        base_security
        * confidence_multiplier
    ).clip(
        lower=0.0,
        upper=1.0,
    )


def calculate_points_per_90(
    players: pd.DataFrame,
) -> pd.Series:
    """Calculate historical FPL points per 90 minutes."""

    points = numeric_series(
        players,
        "total_points",
    )

    minutes = numeric_series(
        players,
        "minutes",
    )

    values = np.where(
        minutes >= 90,
        points / minutes * 90,
        0.0,
    )

    return pd.Series(
        values,
        index=players.index,
        dtype=float,
    ).clip(
        lower=0.0,
        upper=15.0,
    )


def calculate_position_score(
    players: pd.DataFrame,
) -> pd.Series:
    """Build a position-aware historical scoring metric."""

    minutes = numeric_series(
        players,
        "minutes",
    )

    goals = numeric_series(
        players,
        "goals_scored",
    )

    assists = numeric_series(
        players,
        "assists",
    )

    clean_sheets = numeric_series(
        players,
        "clean_sheets",
    )

    saves = numeric_series(
        players,
        "saves",
    )

    bonus = numeric_series(
        players,
        "bonus",
    )

    positions = players["position"].astype(str)

    goal_value = (
        positions
        .map(GOAL_POINTS)
        .fillna(4)
        .astype(float)
    )

    clean_sheet_value = (
        positions
        .map(CLEAN_SHEET_POINTS)
        .fillna(0)
        .astype(float)
    )

    attacking_points = (
        goals * goal_value
        + assists * 3
    )

    clean_sheet_points = (
        clean_sheets
        * clean_sheet_value
    )

    save_points = pd.Series(
        np.where(
            positions == "GKP",
            saves / 3,
            0.0,
        ),
        index=players.index,
        dtype=float,
    )

    role_points = (
        attacking_points
        + clean_sheet_points
        + save_points
        + bonus
    )

    values = np.where(
        minutes >= 90,
        role_points / minutes * 90,
        0.0,
    )

    return pd.Series(
        values,
        index=players.index,
        dtype=float,
    ).clip(
        lower=0.0,
        upper=12.0,
    )


def calculate_fixture_multiplier(
    difficulty: float,
    venue: str,
) -> float:
    """Return the combined difficulty and venue multiplier."""

    if pd.isna(difficulty):
        difficulty_multiplier = 1.0
    else:
        difficulty_value = int(
            round(float(difficulty))
        )

        difficulty_multiplier = (
            FIXTURE_DIFFICULTY_MULTIPLIERS.get(
                difficulty_value,
                1.0,
            )
        )

    if venue == "H":
        venue_multiplier = HOME_MULTIPLIER
    elif venue == "A":
        venue_multiplier = AWAY_MULTIPLIER
    else:
        venue_multiplier = 1.0

    return (
        difficulty_multiplier
        * venue_multiplier
    )


def prepare_fixture_multipliers(
    team_fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Validate fixtures and calculate their multipliers."""

    required_columns = {
        "team_name",
        "event",
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

    fixtures = team_fixtures.copy()

    fixtures["event"] = pd.to_numeric(
        fixtures["event"],
        errors="coerce",
    )

    fixtures = fixtures.dropna(
        subset=[
            "event",
            "team_name",
        ]
    ).copy()

    if fixtures.empty:
        raise ValueError(
            "No usable fixtures are available "
            "for projection calculations."
        )

    fixtures["event"] = (
        fixtures["event"].astype(int)
    )

    fixtures["fixture_multiplier"] = (
        fixtures.apply(
            lambda row: calculate_fixture_multiplier(
                difficulty=row["difficulty"],
                venue=row["venue"],
            ),
            axis=1,
        )
    )

    return fixtures


def build_horizon_fixture_summary(
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise every fixture in the planning horizon."""

    return (
        fixtures
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


def build_next_gameweek_fixture_summary(
    fixtures: pd.DataFrame,
) -> tuple[int, pd.DataFrame]:
    """Summarise only the earliest Gameweek in the horizon."""

    next_gameweek = int(
        fixtures["event"].min()
    )

    next_fixtures = fixtures[
        fixtures["event"] == next_gameweek
    ].copy()

    summary = (
        next_fixtures
        .groupby("team_name")
        .agg(
            next_fixture_count=(
                "event",
                "count",
            ),
            next_fixture_multiplier_total=(
                "fixture_multiplier",
                "sum",
            ),
            next_average_fixture_difficulty=(
                "difficulty",
                "mean",
            ),
        )
        .reset_index()
    )

    return next_gameweek, summary


def build_player_projections(
    players: pd.DataFrame,
    team_fixtures: pd.DataFrame,
    planning_horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """
    Build long-term and next-Gameweek player projections.

    Long-term projections support squad construction.
    Next-Gameweek projections support matchday decisions.
    """

    if planning_horizon <= 0:
        raise ValueError(
            "Planning horizon must be greater than zero."
        )

    projected = players.copy()

    required_columns = {
        "team_name",
        "position",
        "price",
    }

    missing_columns = required_columns.difference(
        projected.columns
    )

    if missing_columns:
        raise ValueError(
            "Player data is missing projection columns: "
            + ", ".join(sorted(missing_columns))
        )

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

    position_score = calculate_position_score(
        projected
    )

    availability = (
        calculate_availability_probability(
            projected
        )
    )

    start_security = calculate_start_security(
        projected
    )

    minutes_security = calculate_minutes_security(
        projected
    )

    raw_points_per_fixture = (
        form * 0.35
        + points_per_game * 0.25
        + points_per_90 * 0.20
        + position_score * 0.20
    )

    projected["availability_probability"] = (
        availability.round(3)
    )

    projected["start_security"] = (
        start_security.round(3)
    )

    projected["minutes_security"] = (
        minutes_security.round(3)
    )

    projected["points_per_90"] = (
        points_per_90.round(2)
    )

    projected["position_score"] = (
        position_score.round(2)
    )

    projected["baseline_points_per_fixture"] = (
        raw_points_per_fixture
        * availability
        * minutes_security
    ).round(3)

    fixtures = prepare_fixture_multipliers(
        team_fixtures
    )

    horizon_summary = (
        build_horizon_fixture_summary(
            fixtures
        )
    )

    next_gameweek, next_summary = (
        build_next_gameweek_fixture_summary(
            fixtures
        )
    )

    projected = projected.merge(
        horizon_summary,
        on="team_name",
        how="left",
    )

    projected = projected.merge(
        next_summary,
        on="team_name",
        how="left",
    )

    projected["next_gameweek"] = (
        next_gameweek
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

    projected["next_fixture_count"] = (
        projected["next_fixture_count"]
        .fillna(0)
        .astype(int)
    )

    projected[
        "next_fixture_multiplier_total"
    ] = (
        projected[
            "next_fixture_multiplier_total"
        ]
        .fillna(0.0)
    )

    projected[
        "next_average_fixture_difficulty"
    ] = (
        projected[
            "next_average_fixture_difficulty"
        ]
        .round(2)
    )

    projected["expected_minutes"] = (
        MINUTES_PER_MATCH
        * projected["minutes_security"]
        * projected["availability_probability"]
        * projected["fixture_count"]
    ).round(1)

    projected["projected_points"] = (
        projected["baseline_points_per_fixture"]
        * projected["fixture_multiplier_total"]
    ).round(2)

    projected["projection_value"] = np.where(
        projected["price"] > 0,
        projected["projected_points"]
        / projected["price"],
        0.0,
    ).round(3)

    projected[
        "next_gameweek_expected_minutes"
    ] = (
        MINUTES_PER_MATCH
        * projected["minutes_security"]
        * projected["availability_probability"]
        * projected["next_fixture_count"]
    ).round(1)

    projected[
        "next_gameweek_projected_points"
    ] = (
        projected["baseline_points_per_fixture"]
        * projected[
            "next_fixture_multiplier_total"
        ]
    ).round(2)

    projected[
        "next_gameweek_projection_value"
    ] = np.where(
        projected["price"] > 0,
        projected[
            "next_gameweek_projected_points"
        ]
        / projected["price"],
        0.0,
    ).round(3)

    return projected