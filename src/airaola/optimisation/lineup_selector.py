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


# ---------------------------------------------------------------------------
# Captaincy reliability policy
# ---------------------------------------------------------------------------
# Raw projected points remain the strongest signal, but captaincy receives
# additional risk controls because doubling the wrong player's score is more
# costly than simply starting the wrong player.
#
# Attackers receive a modest preference because their FPL upside is generally
# less dependent on a clean sheet surviving for 60+ minutes. Defenders remain
# fully eligible and can still win captaincy when their projection advantage is
# genuinely strong. Goalkeepers are heavily discouraged but not hard-blocked.
CAPTAIN_POSITION_MULTIPLIER = {
    "GKP": 0.78,
    "DEF": 0.93,
    "MID": 1.05,
    "FWD": 1.03,
}

VICE_CAPTAIN_POSITION_MULTIPLIER = {
    "GKP": 0.82,
    "DEF": 0.96,
    "MID": 1.03,
    "FWD": 1.02,
}

# Minutes security matters more for captaincy than for ordinary XI selection.
CAPTAIN_MINUTES_SECURITY_POWER = 1.35
VICE_CAPTAIN_MINUTES_SECURITY_POWER = 1.15

# Previous-season reliability is used only as a small confidence modifier.
# The selector remains backwards-compatible when these columns are absent.
CAPTAIN_PRIOR_RELIABILITY_BONUS = 0.08
CAPTAIN_CURRENT_SEASON_EVIDENCE_BONUS = 0.04

# A defender or goalkeeper may still be captain, but only if their raw
# projection is sufficiently better than the strongest MID/FWD alternative.
DEFENDER_CAPTAIN_ADVANTAGE_REQUIRED = 0.75
GOALKEEPER_CAPTAIN_ADVANTAGE_REQUIRED = 1.50


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


def _safe_numeric(
    dataframe: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """Return a safe numeric series without mutating the caller."""

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


def _build_captain_scores(
    starting_xi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build risk-adjusted captain and vice-captain scores.

    Raw next-Gameweek projection stays dominant. Minutes security, position
    and projection reliability act as confidence modifiers rather than
    replacing the underlying points forecast.
    """

    candidates = starting_xi.copy()

    projected_points = _safe_numeric(
        candidates,
        "next_gameweek_projected_points",
    )

    minutes_security = _safe_numeric(
        candidates,
        "minutes_security",
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    prior_reliability = _safe_numeric(
        candidates,
        "prior_reliability",
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    current_season_weight = _safe_numeric(
        candidates,
        "current_season_weight",
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    captain_position_multiplier = (
        candidates["position"]
        .astype(str)
        .map(CAPTAIN_POSITION_MULTIPLIER)
        .fillna(1.0)
        .astype(float)
    )

    vice_position_multiplier = (
        candidates["position"]
        .astype(str)
        .map(VICE_CAPTAIN_POSITION_MULTIPLIER)
        .fillna(1.0)
        .astype(float)
    )

    reliability_multiplier = (
        1.0
        + prior_reliability
        * CAPTAIN_PRIOR_RELIABILITY_BONUS
        + current_season_weight
        * CAPTAIN_CURRENT_SEASON_EVIDENCE_BONUS
    )

    candidates["captain_score"] = (
        projected_points
        * (
            minutes_security
            ** CAPTAIN_MINUTES_SECURITY_POWER
        )
        * captain_position_multiplier
        * reliability_multiplier
    )

    candidates["vice_captain_score"] = (
        projected_points
        * (
            minutes_security
            ** VICE_CAPTAIN_MINUTES_SECURITY_POWER
        )
        * vice_position_multiplier
        * reliability_multiplier
    )

    return candidates


def _apply_position_captain_guard(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stop tiny projection differences from automatically captaining DEF/GKP.

    Defenders and goalkeepers remain eligible when their raw projection clears
    the strongest attacking alternative by a meaningful margin.
    """

    guarded = candidates.copy()

    attacking = guarded[
        guarded["position"].isin(
            [
                "MID",
                "FWD",
            ]
        )
    ]

    if attacking.empty:
        return guarded

    best_attacking_projection = float(
        attacking[
            "next_gameweek_projected_points"
        ].max()
    )

    defender_mask = (
        guarded["position"] == "DEF"
    )

    goalkeeper_mask = (
        guarded["position"] == "GKP"
    )

    defender_allowed = (
        guarded[
            "next_gameweek_projected_points"
        ]
        >= (
            best_attacking_projection
            + DEFENDER_CAPTAIN_ADVANTAGE_REQUIRED
        )
    )

    goalkeeper_allowed = (
        guarded[
            "next_gameweek_projected_points"
        ]
        >= (
            best_attacking_projection
            + GOALKEEPER_CAPTAIN_ADVANTAGE_REQUIRED
        )
    )

    guarded.loc[
        defender_mask & ~defender_allowed,
        "captain_score",
    ] *= 0.80

    guarded.loc[
        goalkeeper_mask & ~goalkeeper_allowed,
        "captain_score",
    ] *= 0.60

    return guarded


def select_captains(
    starting_xi: pd.DataFrame,
) -> tuple[int, int]:
    """
    Select next-Gameweek captain and vice-captain.

    Captaincy uses a risk-adjusted score rather than raw projected points
    alone. MID/FWD players receive a modest upside preference, while DEF/GKP
    must show a clear projection advantage to overcome their lower captaincy
    multiplier.
    """

    if len(starting_xi) < 2:
        raise ValueError(
            "At least two starters are required "
            "for captain selection."
        )

    candidates = _build_captain_scores(
        starting_xi
    )

    candidates = _apply_position_captain_guard(
        candidates
    )

    captain_ranking = candidates.sort_values(
        by=[
            "captain_score",
            "next_gameweek_projected_points",
            "minutes_security",
            "next_gameweek_expected_minutes",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    captain_id = int(
        captain_ranking.iloc[0]["id"]
    )

    vice_candidates = candidates[
        candidates["id"].astype(int)
        != captain_id
    ].copy()

    vice_ranking = vice_candidates.sort_values(
        by=[
            "vice_captain_score",
            "next_gameweek_projected_points",
            "minutes_security",
            "next_gameweek_expected_minutes",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    vice_captain_id = int(
        vice_ranking.iloc[0]["id"]
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
