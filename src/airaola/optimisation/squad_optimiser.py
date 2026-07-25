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
    Prepare players for optimisation.

    Prices are converted from millions into integer tenths because
    CP-SAT works with integer coefficients.
    """

    required_columns = {
        "id",
        "player_name",
        "team_name",
        "position",
        "price",
        "total_points",
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

    pool = pool[
        pool["status"].isin(["a", "d"])
    ].copy()

    pool = pool.dropna(
        subset=[
            "id",
            "team_name",
            "position",
            "price",
            "total_points",
        ]
    )

    pool["price_tenths"] = (
        pool["price"] * 10
    ).round().astype(int)

    pool["optimisation_score"] = (
        pool["total_points"]
        .fillna(0)
        .round()
        .astype(int)
    )

    pool = pool[
        pool["position"].isin(
            POSITION_REQUIREMENTS
        )
    ].copy()

    return pool.reset_index(drop=True)


def optimise_initial_squad(
    players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the legal squad with the highest optimisation score.

    This first version uses current total FPL points as its score.
    Future versions will use projected points instead.
    """

    pool = prepare_optimisation_pool(players)

    model = cp_model.CpModel()

    selected = {
        index: model.new_bool_var(
            f"select_player_{int(player['id'])}"
        )
        for index, player in pool.iterrows()
    }

    # Exactly 15 players
    model.add(
        sum(selected.values()) == SQUAD_SIZE
    )

    # Positional requirements
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

    # Maximum three players per club
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

    # Maximum £100.0m budget
    model.add(
        sum(
            selected[index]
            * int(pool.loc[index, "price_tenths"])
            for index in pool.index
        )
        <= BUDGET_LIMIT_TENTHS
    )

    # Maximise total player score
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
            "Airaola could not find a legal squad."
        )

    selected_indexes = [
        index
        for index in pool.index
        if solver.value(selected[index]) == 1
    ]

    squad = pool.loc[
        selected_indexes
    ].copy()

    squad["selection_score"] = squad[
        "optimisation_score"
    ]

    return squad.reset_index(drop=True)