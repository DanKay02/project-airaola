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


# ---------------------------------------------------------------------------
# Reliability hardening
# ---------------------------------------------------------------------------
# Early in a season, current-season FPL returns are extremely noisy. Airaola
# therefore anchors player performance to a previous-season prior where one is
# available, while gradually allowing the current season to take over.
#
# Current-season minutes are deliberately given a small 1.25x bias so that
# genuinely new information matters, without letting one or two hauls rewrite
# the entire projection model.
CURRENT_SEASON_EVIDENCE_MULTIPLIER = 1.25
CURRENT_SEASON_PRIOR_MINUTES = 600.0
PRIOR_SEASON_RELIABILITY_MINUTES = 900.0

POSITION_BASELINE_POINTS = {
    "GKP": 3.2,
    "DEF": 3.2,
    "MID": 3.8,
    "FWD": 3.8,
}

# Used only when evidence about a player's future minutes is still scarce.
# The prior is intentionally fairly generous: two early Gameweeks should not
# make every regular starter look like a 55-minute player.
POSITION_MINUTES_SECURITY_PRIOR = {
    "GKP": 0.90,
    "DEF": 0.82,
    "MID": 0.82,
    "FWD": 0.82,
}

MINUTES_SECURITY_EVIDENCE_STARTS = 6.0


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
    """
    Estimate future minutes security.

    Current starts and involvement remain the main evidence, but tiny early-
    season samples are shrunk toward a sensible positional prior rather than
    multiplying every player's security down toward zero.
    """

    start_security = calculate_start_security(
        players
    )

    involvement = calculate_minutes_involvement(
        players
    )

    starts = numeric_series(
        players,
        "starts",
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

    observed_security = pd.Series(
        np.where(
            positions == "GKP",
            goalkeeper_security,
            outfield_security,
        ),
        index=players.index,
        dtype=float,
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    prior_security = (
        positions
        .map(POSITION_MINUTES_SECURITY_PRIOR)
        .fillna(0.80)
        .astype(float)
    )

    evidence_weight = np.sqrt(
        starts.clip(lower=0.0)
        / MINUTES_SECURITY_EVIDENCE_STARTS
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    security = (
        prior_security
        * (1.0 - evidence_weight)
        + observed_security
        * evidence_weight
    )

    return pd.Series(
        security,
        index=players.index,
        dtype=float,
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



def calculate_prior_points_per_90(
    players: pd.DataFrame,
) -> pd.Series:
    """Calculate previous-season FPL points per 90 minutes."""

    prior_points = numeric_series(
        players,
        "prior_total_points",
    )

    prior_minutes = numeric_series(
        players,
        "prior_minutes",
    )

    values = np.where(
        prior_minutes >= 90,
        prior_points / prior_minutes * 90,
        np.nan,
    )

    return pd.Series(
        values,
        index=players.index,
        dtype=float,
    ).clip(
        lower=0.0,
        upper=12.0,
    )


def calculate_previous_season_prior(
    players: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Build a stable prior from previous-season FPL performance.

    Returns:
    - prior points per fixture
    - prior reliability
    - whether usable prior history exists

    Players with little or no previous-season evidence are shrunk heavily
    toward a positional baseline.
    """

    positions = players["position"].astype(str)

    position_baseline = (
        positions
        .map(POSITION_BASELINE_POINTS)
        .fillna(3.5)
        .astype(float)
    )

    prior_minutes = numeric_series(
        players,
        "prior_minutes",
    )

    prior_points_per_90 = (
        calculate_prior_points_per_90(
            players
        )
    )

    if "prior_history_available" in players.columns:
        prior_available = (
            players["prior_history_available"]
            .fillna(False)
            .astype(bool)
        )
    else:
        prior_available = pd.Series(
            False,
            index=players.index,
            dtype=bool,
        )

    usable_prior = (
        prior_available
        & prior_minutes.ge(90)
        & prior_points_per_90.notna()
    )

    prior_reliability = (
        prior_minutes.clip(lower=0.0)
        / (
            prior_minutes.clip(lower=0.0)
            + PRIOR_SEASON_RELIABILITY_MINUTES
        )
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    prior_reliability = (
        prior_reliability
        * usable_prior.astype(float)
    )

    previous_season_prior = (
        position_baseline
        * (1.0 - prior_reliability)
        + prior_points_per_90.fillna(
            position_baseline
        )
        * prior_reliability
    )

    return (
        previous_season_prior.clip(
            lower=1.0,
            upper=8.0,
        ),
        prior_reliability,
        usable_prior,
    )


def calculate_current_season_weight(
    players: pd.DataFrame,
) -> pd.Series:
    """
    Return how much current-season evidence should influence projections.

    At 180 current-season minutes the weight is still deliberately modest.
    As the season matures, current evidence gradually becomes dominant.
    """

    current_minutes = numeric_series(
        players,
        "minutes",
    ).clip(
        lower=0.0,
    )

    effective_minutes = (
        current_minutes
        * CURRENT_SEASON_EVIDENCE_MULTIPLIER
    )

    weight = (
        effective_minutes
        / (
            effective_minutes
            + CURRENT_SEASON_PRIOR_MINUTES
        )
    )

    return pd.Series(
        weight,
        index=players.index,
        dtype=float,
    ).clip(
        lower=0.0,
        upper=1.0,
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

    fixtures["difficulty"] = pd.to_numeric(
        fixtures["difficulty"],
        errors="coerce",
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


def build_gameweek_fixture_summary(
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise fixtures separately for every team and Gameweek."""

    return (
        fixtures
        .groupby(
            [
                "team_name",
                "event",
            ]
        )
        .agg(
            gameweek_fixture_count=(
                "event",
                "count",
            ),
            gameweek_fixture_multiplier_total=(
                "fixture_multiplier",
                "sum",
            ),
            gameweek_average_fixture_difficulty=(
                "difficulty",
                "mean",
            ),
        )
        .reset_index()
    )


def add_gameweek_projection_columns(
    projected: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    """Add fixture and projection columns for every Gameweek."""

    gameweek_summary = (
        build_gameweek_fixture_summary(
            fixtures
        )
    )

    gameweeks = sorted(
        int(event)
        for event in gameweek_summary[
            "event"
        ].dropna().unique()
    )

    result = projected.copy()

    for gameweek in gameweeks:
        gameweek_rows = gameweek_summary[
            gameweek_summary["event"]
            == gameweek
        ][
            [
                "team_name",
                "gameweek_fixture_count",
                "gameweek_fixture_multiplier_total",
                "gameweek_average_fixture_difficulty",
            ]
        ].copy()

        fixture_count_column = (
            f"gw_{gameweek}_fixture_count"
        )

        multiplier_column = (
            f"gw_{gameweek}_fixture_multiplier_total"
        )

        difficulty_column = (
            f"gw_{gameweek}_average_fixture_difficulty"
        )

        expected_minutes_column = (
            f"gw_{gameweek}_expected_minutes"
        )

        projected_points_column = (
            f"gw_{gameweek}_projected_points"
        )

        gameweek_rows = gameweek_rows.rename(
            columns={
                "gameweek_fixture_count":
                    fixture_count_column,
                "gameweek_fixture_multiplier_total":
                    multiplier_column,
                "gameweek_average_fixture_difficulty":
                    difficulty_column,
            }
        )

        result = result.merge(
            gameweek_rows,
            on="team_name",
            how="left",
        )

        result[fixture_count_column] = (
            result[fixture_count_column]
            .fillna(0)
            .astype(int)
        )

        result[multiplier_column] = (
            result[multiplier_column]
            .fillna(0.0)
        )

        result[difficulty_column] = (
            result[difficulty_column]
            .round(2)
        )

        result[expected_minutes_column] = (
            MINUTES_PER_MATCH
            * result["minutes_security"]
            * result["availability_probability"]
            * result[fixture_count_column]
        ).round(1)

        result[projected_points_column] = (
            result["baseline_points_per_fixture"]
            * result[multiplier_column]
        ).round(2)

    return result, gameweeks


def build_player_projections(
    players: pd.DataFrame,
    team_fixtures: pd.DataFrame,
    planning_horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """
    Build long-term and per-Gameweek player projections.

    Long-term projections support squad construction.
    Per-Gameweek projections support captaincy and matchday decisions.
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

    position_baseline = (
        projected["position"]
        .astype(str)
        .map(POSITION_BASELINE_POINTS)
        .fillna(3.5)
        .astype(float)
    )

    observed_points_per_fixture = (
        form * 0.25
        + points_per_game * 0.25
        + points_per_90 * 0.25
        + position_score * 0.25
    ).clip(
        lower=0.0,
        upper=10.0,
    )

    (
        previous_season_prior,
        prior_reliability,
        prior_history_used,
    ) = calculate_previous_season_prior(
        projected
    )

    current_season_weight = (
        calculate_current_season_weight(
            projected
        )
    )

    raw_points_per_fixture = (
        previous_season_prior
        * (1.0 - current_season_weight)
        + observed_points_per_fixture
        * current_season_weight
    ).clip(
        lower=1.0,
        upper=8.0,
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

    projected["position_baseline"] = (
        position_baseline.round(2)
    )

    projected["observed_points_per_fixture"] = (
        observed_points_per_fixture.round(3)
    )

    projected["previous_season_prior"] = (
        previous_season_prior.round(3)
    )

    projected["prior_reliability"] = (
        prior_reliability.round(3)
    )

    projected["prior_history_used"] = (
        prior_history_used.astype(bool)
    )

    projected["current_season_weight"] = (
        current_season_weight.round(3)
    )

    projected["baseline_points_per_fixture"] = (
        raw_points_per_fixture
        * availability
        * minutes_security
    ).round(3)

    fixtures = prepare_fixture_multipliers(
        team_fixtures
    )

    available_gameweeks = sorted(
        int(event)
        for event in fixtures["event"].unique()
    )

    selected_gameweeks = (
        available_gameweeks[:planning_horizon]
    )

    fixtures = fixtures[
        fixtures["event"].isin(
            selected_gameweeks
        )
    ].copy()

    horizon_summary = (
        build_horizon_fixture_summary(
            fixtures
        )
    )

    projected = projected.merge(
        horizon_summary,
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

    projected, gameweeks = (
        add_gameweek_projection_columns(
            projected=projected,
            fixtures=fixtures,
        )
    )

    if not gameweeks:
        raise ValueError(
            "No Gameweeks are available for "
            "per-Gameweek projections."
        )

    next_gameweek = int(gameweeks[0])

    projected["next_gameweek"] = (
        next_gameweek
    )

    projected["next_fixture_count"] = (
        projected[
            f"gw_{next_gameweek}_fixture_count"
        ]
    )

    projected[
        "next_fixture_multiplier_total"
    ] = projected[
        f"gw_{next_gameweek}_fixture_multiplier_total"
    ]

    projected[
        "next_average_fixture_difficulty"
    ] = projected[
        f"gw_{next_gameweek}_average_fixture_difficulty"
    ]

    projected[
        "next_gameweek_expected_minutes"
    ] = projected[
        f"gw_{next_gameweek}_expected_minutes"
    ]

    projected[
        "next_gameweek_projected_points"
    ] = projected[
        f"gw_{next_gameweek}_projected_points"
    ]

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