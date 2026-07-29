from __future__ import annotations

import pandas as pd
from ortools.sat.python import cp_model


STARTING_XI_SIZE = 11

MIN_STARTERS_BY_POSITION = {
    "GKP": 1,
    "DEF": 3,
    "MID": 2,
    "FWD": 1,
}

MAX_STARTERS_BY_POSITION = {
    "GKP": 1,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}


def validate_squad_for_lineup(
    squad: pd.DataFrame,
) -> None:
    """Confirm the squad contains matchday projection data."""

    required_columns = {
        "id",
        "player_name",
        "team_name",
        "position",
        "minutes_security",
        "next_gameweek",
        "next_fixture_count",
        "next_gameweek_expected_minutes",
        "next_gameweek_projected_points",
    }

    missing_columns = required_columns.difference(
        squad.columns
    )

    if missing_columns:
        raise ValueError(
            "Squad data is missing lineup columns: "
            + ", ".join(sorted(missing_columns))
        )

    if len(squad) != 15:
        raise ValueError(
            "Starting-lineup selection requires a "
            "complete 15-player squad."
        )


def prepare_lineup_pool(
    squad: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare next-Gameweek scores for lineup optimisation."""

    pool = squad.copy()

    numeric_columns = [
        "minutes_security",
        "next_fixture_count",
        "next_gameweek_expected_minutes",
        "next_gameweek_projected_points",
    ]

    for column in numeric_columns:
        pool[column] = pd.to_numeric(
            pool[column],
            errors="coerce",
        ).fillna(0.0)

    pool["lineup_score"] = (
        pool["next_gameweek_projected_points"]
        * 0.85
        + pool["minutes_security"]
        * 0.75
        + (
            pool["next_gameweek_expected_minutes"]
            / 90
        )
        * 0.25
    )

    pool["lineup_score_integer"] = (
        pool["lineup_score"]
        .mul(1000)
        .round()
        .astype(int)
    )

    return pool.reset_index(drop=True)


def select_starting_xi(
    squad: pd.DataFrame,
) -> pd.DataFrame:
    """Select the strongest legal next-Gameweek XI."""

    validate_squad_for_lineup(squad)

    pool = prepare_lineup_pool(squad)

    model = cp_model.CpModel()

    selected = {
        index: model.new_bool_var(
            f"start_player_{int(player['id'])}"
        )
        for index, player in pool.iterrows()
    }

    model.add(
        sum(selected.values())
        == STARTING_XI_SIZE
    )

    for position in MIN_STARTERS_BY_POSITION:
        position_indexes = pool.index[
            pool["position"] == position
        ].tolist()

        model.add(
            sum(
                selected[index]
                for index in position_indexes
            )
            >= MIN_STARTERS_BY_POSITION[position]
        )

        model.add(
            sum(
                selected[index]
                for index in position_indexes
            )
            <= MAX_STARTERS_BY_POSITION[position]
        )

    model.maximize(
        sum(
            selected[index]
            * int(
                pool.loc[
                    index,
                    "lineup_score_integer",
                ]
            )
            for index in pool.index
        )
    )

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 8

    status = solver.solve(model)

    if status not in {
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    }:
        raise RuntimeError(
            "Project Airaola could not find "
            "a legal starting XI."
        )

    selected_indexes = [
        index
        for index in pool.index
        if solver.value(selected[index]) == 1
    ]

    starting_xi = pool.loc[
        selected_indexes
    ].copy()

    starting_xi["is_starter"] = True

    return starting_xi.reset_index(drop=True)


def select_captains(
    starting_xi: pd.DataFrame,
) -> tuple[int, int]:
    """Select next-Gameweek captain and vice-captain."""

    candidates = starting_xi.copy()

    candidates["captain_score"] = (
        candidates[
            "next_gameweek_projected_points"
        ]
        * candidates["minutes_security"]
    )

    candidates = candidates.sort_values(
        by=[
            "captain_score",
            "next_gameweek_projected_points",
            "next_gameweek_expected_minutes",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    if len(candidates) < 2:
        raise ValueError(
            "At least two starters are required "
            "for captain selection."
        )

    captain_id = int(
        candidates.iloc[0]["id"]
    )

    vice_captain_id = int(
        candidates.iloc[1]["id"]
    )

    return captain_id, vice_captain_id


def build_bench(
    squad: pd.DataFrame,
    starting_xi: pd.DataFrame,
) -> pd.DataFrame:
    """Rank substitutes using next-Gameweek projections."""

    starter_ids = set(
        starting_xi["id"].astype(int)
    )

    bench = squad[
        ~squad["id"].astype(int).isin(
            starter_ids
        )
    ].copy()

    reserve_goalkeepers = bench[
        bench["position"] == "GKP"
    ].copy()

    outfield_bench = bench[
        bench["position"] != "GKP"
    ].copy()

    outfield_bench["bench_score"] = (
        outfield_bench[
            "next_gameweek_projected_points"
        ]
        * outfield_bench["minutes_security"]
    )

    outfield_bench = outfield_bench.sort_values(
        by=[
            "bench_score",
            "next_gameweek_projected_points",
            "next_gameweek_expected_minutes",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    outfield_bench["bench_order"] = range(
        1,
        len(outfield_bench) + 1,
    )

    reserve_goalkeepers["bench_order"] = 4

    return pd.concat(
        [
            outfield_bench,
            reserve_goalkeepers,
        ],
        ignore_index=True,
    )


def select_gameweek_team(
    squad: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
    int,
]:
    """Select next-GW XI, bench and captaincy."""

    starting_xi = select_starting_xi(
        squad
    )

    captain_id, vice_captain_id = (
        select_captains(starting_xi)
    )

    starting_xi["is_captain"] = (
        starting_xi["id"].astype(int)
        == captain_id
    )

    starting_xi["is_vice_captain"] = (
        starting_xi["id"].astype(int)
        == vice_captain_id
    )

    bench = build_bench(
        squad=squad,
        starting_xi=starting_xi,
    )

    return (
        starting_xi,
        bench,
        captain_id,
        vice_captain_id,
    )