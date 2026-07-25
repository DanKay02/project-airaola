from __future__ import annotations

import pandas as pd
from ortools.sat.python import cp_model


SQUAD_SIZE = 15
BUDGET_LIMIT_TENTHS = 1000
MAX_PLAYERS_PER_CLUB = 3

POSITION_REQUIREMENTS = {
    "GKP": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}


def prepare_optimisation_pool(
    players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare player data for squad optimisation.

    Prices are converted into integer tenths because OR-Tools CP-SAT
    requires integer coefficients.

    Projected points are multiplied by 100 so decimal projections can
    still be optimised accurately.
    """

    required_columns = {
        "id",
        "player_name",
        "team_name",
        "position",
        "price",
        "projected_points",
        "status",
    }

    missing_columns = required_columns.difference(
        players.columns
    )

    if missing_columns:
        raise ValueError(
            "Player data is missing required optimiser columns: "
            + ", ".join(sorted(missing_columns))
        )

    pool = players.copy()

    # Keep players marked as available or doubtful.
    pool = pool[
        pool["status"].isin(["a", "d"])
    ].copy()

    pool = pool.dropna(
        subset=[
            "id",
            "player_name",
            "team_name",
            "position",
            "price",
            "projected_points",
        ]
    )

    pool = pool[
        pool["position"].isin(
            POSITION_REQUIREMENTS
        )
    ].copy()

    pool["price_tenths"] = (
        pd.to_numeric(
            pool["price"],
            errors="coerce",
        )
        .mul(10)
        .round()
        .astype(int)
    )

    pool["optimisation_score"] = (
        pd.to_numeric(
            pool["projected_points"],
            errors="coerce",
        )
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

    return pool.reset_index(drop=True)


def validate_position_pool(
    pool: pd.DataFrame,
) -> None:
    """
    Confirm enough eligible players exist in every position.

    This produces a clearer error before the optimiser is started.
    """

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
    Select the legal 15-player squad with the highest projected score.

    Constraints:
    - exactly 15 players
    - exactly 2 GKP
    - exactly 5 DEF
    - exactly 5 MID
    - exactly 3 FWD
    - maximum 3 players per club
    - maximum £100.0m total cost
    """

    pool = prepare_optimisation_pool(players)
    validate_position_pool(pool)

    model = cp_model.CpModel()

    selected = {
        index: model.new_bool_var(
            f"select_player_{int(player['id'])}"
        )
        for index, player in pool.iterrows()
    }

    # Exactly 15 players.
    model.add(
        sum(selected.values()) == SQUAD_SIZE
    )

    # Exact positional composition.
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

    # Maximum three players from any one club.
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

    # Total cost cannot exceed £100.0m.
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

    # Maximise projected points.
    model.maximize(
        sum(
            selected[index]
            * int(
                pool.loc[
                    index,
                    "optimisation_score",
                ]
            )
            for index in pool.index
        )
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
            "a legal projected squad."
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

    return squad.reset_index(drop=True)