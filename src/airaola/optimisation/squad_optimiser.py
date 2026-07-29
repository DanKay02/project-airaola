from __future__ import annotations

import re

import pandas as pd
from ortools.sat.python import cp_model


SQUAD_SIZE = 15
BUDGET_LIMIT_TENTHS = 1000
MAX_PLAYERS_PER_CLUB = 3

MIN_MINUTES_SECURITY = 0.35
MIN_EXPECTED_MINUTES = 135.0
MIN_GOALKEEPER_SECURITY = 0.65

POSITION_REQUIREMENTS = {
    "GKP": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}

GAMEWEEK_POINTS_PATTERN = re.compile(
    r"^gw_(\d+)_projected_points$"
)


def discover_gameweek_projection_columns(
    players: pd.DataFrame,
) -> dict[int, str]:
    """Find per-Gameweek projection columns."""

    discovered: dict[int, str] = {}

    for column in players.columns:
        match = GAMEWEEK_POINTS_PATTERN.match(
            str(column)
        )

        if match:
            gameweek = int(match.group(1))
            discovered[gameweek] = str(column)

    if not discovered:
        raise ValueError(
            "Player data does not contain any "
            "per-Gameweek projection columns."
        )

    return dict(
        sorted(discovered.items())
    )


def prepare_optimisation_pool(
    players: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[int, str],
]:
    """
    Prepare player data for captaincy-aware squad optimisation.

    The normal projected score rewards total squad strength across the
    planning horizon. Separate per-Gameweek scores allow the optimiser
    to add one captain bonus in every Gameweek.
    """

    required_columns = {
        "id",
        "player_name",
        "team_name",
        "position",
        "price",
        "projected_points",
        "status",
        "minutes_security",
        "expected_minutes",
    }

    missing_columns = required_columns.difference(
        players.columns
    )

    if missing_columns:
        raise ValueError(
            "Player data is missing required optimiser columns: "
            + ", ".join(sorted(missing_columns))
        )

    gameweek_columns = (
        discover_gameweek_projection_columns(
            players
        )
    )

    pool = players.copy()

    numeric_columns = [
        "minutes_security",
        "expected_minutes",
        "price",
        "projected_points",
        *gameweek_columns.values(),
    ]

    for column in numeric_columns:
        pool[column] = pd.to_numeric(
            pool[column],
            errors="coerce",
        )

    pool = pool[
        pool["status"].isin(["a", "d"])
        & (
            pool["minutes_security"]
            >= MIN_MINUTES_SECURITY
        )
        & (
            pool["expected_minutes"]
            >= MIN_EXPECTED_MINUTES
        )
    ].copy()

    pool = pool[
        (
            pool["position"] != "GKP"
        )
        | (
            pool["minutes_security"]
            >= MIN_GOALKEEPER_SECURITY
        )
    ].copy()

    pool = pool.dropna(
        subset=[
            "id",
            "player_name",
            "team_name",
            "position",
            "price",
            "projected_points",
            "minutes_security",
            "expected_minutes",
        ]
    )

    pool = pool[
        pool["position"].isin(
            POSITION_REQUIREMENTS
        )
    ].copy()

    pool["price_tenths"] = (
        pool["price"]
        .mul(10)
        .round()
        .astype(int)
    )

    pool["optimisation_score"] = (
        pool["projected_points"]
        .fillna(0.0)
        .mul(100)
        .round()
        .astype(int)
    )

    for gameweek, column in (
        gameweek_columns.items()
    ):
        score_column = (
            f"gw_{gameweek}_captain_score"
        )

        pool[score_column] = (
            pool[column]
            .fillna(0.0)
            .mul(100)
            .round()
            .astype(int)
        )

    pool = pool[
        pool["price_tenths"] > 0
    ].copy()

    if pool.empty:
        raise ValueError(
            "No eligible players remain after preparing "
            "the optimisation pool."
        )

    return (
        pool.reset_index(drop=True),
        gameweek_columns,
    )


def validate_position_pool(
    pool: pd.DataFrame,
) -> None:
    """Confirm enough eligible players exist in every position."""

    for position, required_count in (
        POSITION_REQUIREMENTS.items()
    ):
        available_count = int(
            (pool["position"] == position).sum()
        )

        if available_count < required_count:
            raise ValueError(
                f"Not enough eligible {position} players. "
                f"Required: {required_count}. "
                f"Available: {available_count}."
            )


def optimise_initial_squad(
    players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select a legal, captaincy-aware 15-player squad.

    The objective contains:
    - every selected player's total planning-horizon projection
    - one additional projected score for the best captain in each
      individual Gameweek

    This reflects the FPL rule that the captain's points are doubled.
    """

    pool, gameweek_columns = (
        prepare_optimisation_pool(players)
    )

    validate_position_pool(pool)

    model = cp_model.CpModel()

    selected = {
        index: model.new_bool_var(
            f"select_player_{int(player['id'])}"
        )
        for index, player in pool.iterrows()
    }

    captain = {
        (
            gameweek,
            index,
        ): model.new_bool_var(
            "captain_"
            f"gw_{gameweek}_"
            f"player_{int(pool.loc[index, 'id'])}"
        )
        for gameweek in gameweek_columns
        for index in pool.index
    }

    model.add(
        sum(selected.values())
        == SQUAD_SIZE
    )

    for position, required_count in (
        POSITION_REQUIREMENTS.items()
    ):
        position_indexes = pool.index[
            pool["position"] == position
        ].tolist()

        model.add(
            sum(
                selected[index]
                for index in position_indexes
            )
            == required_count
        )

    for club_name in sorted(
        pool["team_name"].unique()
    ):
        club_indexes = pool.index[
            pool["team_name"] == club_name
        ].tolist()

        model.add(
            sum(
                selected[index]
                for index in club_indexes
            )
            <= MAX_PLAYERS_PER_CLUB
        )

    model.add(
        sum(
            selected[index]
            * int(
                pool.loc[
                    index,
                    "price_tenths",
                ]
            )
            for index in pool.index
        )
        <= BUDGET_LIMIT_TENTHS
    )

    for gameweek in gameweek_columns:
        model.add(
            sum(
                captain[
                    (
                        gameweek,
                        index,
                    )
                ]
                for index in pool.index
            )
            == 1
        )

        for index in pool.index:
            model.add(
                captain[
                    (
                        gameweek,
                        index,
                    )
                ]
                <= selected[index]
            )

    squad_score = sum(
        selected[index]
        * int(
            pool.loc[
                index,
                "optimisation_score",
            ]
        )
        for index in pool.index
    )

    captaincy_score = sum(
        captain[
            (
                gameweek,
                index,
            )
        ]
        * int(
            pool.loc[
                index,
                f"gw_{gameweek}_captain_score",
            ]
        )
        for gameweek in gameweek_columns
        for index in pool.index
    )

    model.maximize(
        squad_score
        + captaincy_score
    )

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 8

    status = solver.solve(model)

    valid_statuses = {
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    }

    if status not in valid_statuses:
        raise RuntimeError(
            "Project Airaola could not find "
            "a legal captaincy-aware squad."
        )

    selected_indexes = [
        index
        for index in pool.index
        if solver.value(selected[index]) == 1
    ]

    squad = pool.loc[
        selected_indexes
    ].copy()

    squad["selection_score"] = (
        squad["optimisation_score"] / 100
    )

    squad["selected_by_optimiser"] = True

    captaincy_gameweeks: dict[int, list[int]] = {
        int(player_id): []
        for player_id in squad["id"]
    }

    for gameweek in gameweek_columns:
        for index in selected_indexes:
            if solver.value(
                captain[
                    (
                        gameweek,
                        index,
                    )
                ]
            ) == 1:
                player_id = int(
                    pool.loc[index, "id"]
                )

                captaincy_gameweeks[
                    player_id
                ].append(gameweek)

    squad["projected_captain_gameweeks"] = (
        squad["id"]
        .astype(int)
        .map(
            lambda player_id: ",".join(
                str(gameweek)
                for gameweek in (
                    captaincy_gameweeks.get(
                        player_id,
                        [],
                    )
                )
            )
        )
    )

    squad["captaincy_appearances"] = (
        squad["id"]
        .astype(int)
        .map(
            lambda player_id: len(
                captaincy_gameweeks.get(
                    player_id,
                    [],
                )
            )
        )
        .astype(int)
    )

    return squad.reset_index(drop=True)