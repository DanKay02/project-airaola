from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_GAMEWEEK_MINUTES = 90
DEFAULT_HORIZON = 5


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """Convert an FPL column into a safe numeric series."""

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

    chance_column = numeric_series(
        players,
        "chance_of_playing_next_round",
        default=np.nan,
    )

    chance_probability = chance_column / 100

    return chance_probability.fillna(
        status_probability
    ).clip(lower=0.0, upper=1.0)


def calculate_minutes_reliability(
    players: pd.DataFrame,
) -> pd.Series:
    """
    Estimate how reliably a player receives significant minutes.

    This baseline uses season minutes and appearances where available.
    """

    minutes = numeric_series(players, "minutes")
    starts = numeric_series(players, "starts")

    estimated_appearances = np.maximum(
        starts,
        np.ceil(minutes / 90),
    )

    average_minutes = np.where(
        estimated_appearances > 0,
        minutes / estimated_appearances,
        0.0,
    )

    reliability = average_minutes / 90

    return pd.Series(
        reliability,
        index=players.index,
    ).clip(lower=0.0, upper=1.0)


def calculate_points_per_90(
    players: pd.DataFrame,
) -> pd.Series:
    """Calculate historic FPL points scored per 90 minutes."""

    points = numeric_series(players, "total_points")
    minutes = numeric_series(players, "minutes")

    points_per_90 = np.where(
        minutes >= 90,
        points / minutes * 90,
        0.0,
    )

    return pd.Series(
        points_per_90,
        index=players.index,
    ).clip(lower=0.0, upper=15.0)


def build_player_projections(
    players: pd.DataFrame,
    planning_horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """
    Produce baseline forward projections for every player.

    This version does not yet account for individual fixtures.
    """

    if planning_horizon <= 0:
        raise ValueError(
            "Planning horizon must be greater than zero."
        )

    projected = players.copy()

    form = numeric_series(projected, "form")
    points_per_game = numeric_series(
        projected,
        "points_per_game",
    )
    points_per_90 = calculate_points_per_90(projected)

    availability = calculate_availability_probability(
        projected
    )

    minutes_reliability = calculate_minutes_reliability(
        projected
    )

    expected_minutes_per_gameweek = (
        DEFAULT_GAMEWEEK_MINUTES
        * minutes_reliability
        * availability
    )

    expected_minutes = (
        expected_minutes_per_gameweek
        * planning_horizon
    )

    raw_points_per_game = (
        form * 0.45
        + points_per_game * 0.30
        + points_per_90 * 0.25
    )

    projected_points_per_gameweek = (
        raw_points_per_game
        * minutes_reliability
        * availability
    )

    projected_points = (
        projected_points_per_gameweek
        * planning_horizon
    )

    projected["availability_probability"] = (
        availability.round(3)
    )

    projected["minutes_reliability"] = (
        minutes_reliability.round(3)
    )

    projected["expected_minutes"] = (
        expected_minutes.round(1)
    )

    projected["points_per_90"] = (
        points_per_90.round(2)
    )

    projected["projected_points_per_gameweek"] = (
        projected_points_per_gameweek.round(2)
    )

    projected["projected_points"] = (
        projected_points.round(2)
    )

    projected["projection_value"] = np.where(
        projected["price"] > 0,
        projected["projected_points"]
        / projected["price"],
        0.0,
    ).round(3)

    return projected