from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
from ortools.sat.python import cp_model

from airaola.strategy.chip_strategy import (
    FREE_HIT,
    WILDCARD,
    SquadChipEvaluation,
)


SQUAD_SIZE = 15
STARTING_XI_SIZE = 11
DEFAULT_BUDGET_LIMIT = 100.0
MAX_PLAYERS_PER_CLUB = 3

MIN_MINUTES_SECURITY = 0.35
MIN_EXPECTED_MINUTES = 135.0
MIN_GOALKEEPER_SECURITY = 0.55

SECURE_PLAYER_THRESHOLD = 0.70
BENCH_COVER_WEIGHT = 0.15

SOLVER_TIME_LIMIT_SECONDS = 30.0
SOLVER_WORKERS = 8

POSITION_REQUIREMENTS = {
    "GKP": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}

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

GAMEWEEK_POINTS_PATTERN = re.compile(
    r"^gw_(\d+)_projected_points$"
)


@dataclass(frozen=True)
class SolvedSquad:
    """Store one completed lineup-aware optimisation result."""

    squad: pd.DataFrame

    gameweeks: tuple[int, ...]

    total_objective_points: float
    starting_points: float
    captain_points: float
    bench_cover_points: float

    squad_cost: float
    bank_remaining: float

    secure_player_count: int


def discover_gameweek_projection_columns(
    players: pd.DataFrame,
) -> dict[int, str]:
    """Find every per-Gameweek projection column."""

    discovered: dict[int, str] = {}

    for column in players.columns:
        match = GAMEWEEK_POINTS_PATTERN.match(
            str(column)
        )

        if match:
            gameweek = int(
                match.group(1)
            )

            discovered[gameweek] = str(
                column
            )

    if not discovered:
        raise ValueError(
            "Player data does not contain any "
            "per-Gameweek projection columns."
        )

    return dict(
        sorted(
            discovered.items()
        )
    )


def _safe_numeric(
    dataframe: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """Return one safely converted numeric column."""

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


def _normalise_budget(
    available_budget: float,
) -> float:
    """Validate and normalise an FPL squad budget."""

    try:
        parsed_budget = float(
            available_budget
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Available squad budget must be numeric."
        ) from error

    if parsed_budget <= 0:
        raise ValueError(
            "Available squad budget must be positive."
        )

    return round(
        parsed_budget,
        1,
    )


def _budget_to_tenths(
    available_budget: float,
) -> int:
    """Convert an FPL budget into integer tenths."""

    return int(
        round(
            _normalise_budget(
                available_budget
            )
            * 10
        )
    )


def prepare_optimisation_pool(
    players: pd.DataFrame,
    preserve_player_ids: set[int] | None = None,
) -> tuple[
    pd.DataFrame,
    dict[int, str],
]:
    """
    Prepare player data for lineup-aware optimisation.

    Players that fail normal availability or minutes filters are
    excluded unless their IDs are explicitly preserved. Preserved
    players allow Airaola to evaluate the actual current squad,
    including injured, doubtful or low-security players.
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
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    gameweek_columns = (
        discover_gameweek_projection_columns(
            players
        )
    )

    pool = players.copy()

    numeric_columns = [
        "id",
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
    ).copy()

    pool["id"] = (
        pool["id"]
        .astype(int)
    )

    preserved_ids = {
        int(
            player_id
        )
        for player_id in (
            preserve_player_ids
            or set()
        )
    }

    normally_eligible = (
        pool["status"].isin(
            [
                "a",
                "d",
            ]
        )
        & (
            pool["minutes_security"]
            >= MIN_MINUTES_SECURITY
        )
        & (
            pool["expected_minutes"]
            >= MIN_EXPECTED_MINUTES
        )
        & (
            (
                pool["position"]
                != "GKP"
            )
            | (
                pool["minutes_security"]
                >= MIN_GOALKEEPER_SECURITY
            )
        )
    )

    explicitly_preserved = (
        pool["id"].isin(
            preserved_ids
        )
    )

    pool = pool[
        normally_eligible
        | explicitly_preserved
    ].copy()

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

    for (
        gameweek,
        column,
    ) in gameweek_columns.items():
        score_column = (
            f"gw_{gameweek}_optimisation_score"
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

    pool = (
        pool.drop_duplicates(
            subset=[
                "id",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    if pool.empty:
        raise ValueError(
            "No eligible players remain after preparing "
            "the optimisation pool."
        )

    return (
        pool,
        gameweek_columns,
    )


def validate_position_pool(
    pool: pd.DataFrame,
) -> None:
    """Confirm enough eligible players exist in every position."""

    for (
        position,
        required_count,
    ) in POSITION_REQUIREMENTS.items():
        available_count = int(
            (
                pool["position"]
                == position
            ).sum()
        )

        if available_count < required_count:
            raise ValueError(
                f"Not enough eligible {position} players. "
                f"Required: {required_count}. "
                f"Available: {available_count}."
            )


def _resolve_gameweeks(
    gameweek_columns: dict[int, str],
    selected_gameweeks: list[int] | tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Resolve and validate the Gameweeks used by an optimiser."""

    available_gameweeks = tuple(
        gameweek_columns.keys()
    )

    if selected_gameweeks is None:
        return available_gameweeks

    resolved = tuple(
        sorted(
            {
                int(
                    gameweek
                )
                for gameweek
                in selected_gameweeks
            }
        )
    )

    if not resolved:
        raise ValueError(
            "At least one optimisation Gameweek is required."
        )

    missing_gameweeks = set(
        resolved
    ).difference(
        gameweek_columns
    )

    if missing_gameweeks:
        raise ValueError(
            "Projection data is missing requested Gameweeks: "
            + ", ".join(
                str(
                    gameweek
                )
                for gameweek
                in sorted(
                    missing_gameweeks
                )
            )
        )

    return resolved


def _build_squad_annotations(
    pool: pd.DataFrame,
    selected_indexes: list[int],
    gameweeks: tuple[int, ...],
    solver: cp_model.CpSolver,
    starter: dict[tuple[int, int], cp_model.IntVar],
    captain: dict[tuple[int, int], cp_model.IntVar],
    vice_captain: dict[
        tuple[int, int],
        cp_model.IntVar,
    ],
) -> pd.DataFrame:
    """Attach lineup, captaincy and selection metadata."""

    squad = pool.loc[
        selected_indexes
    ].copy()

    squad["selection_score"] = (
        squad["optimisation_score"]
        / 100
    )

    squad["selected_by_optimiser"] = True

    starter_gameweeks: dict[int, list[int]] = {
        int(
            player_id
        ): []
        for player_id in squad["id"]
    }

    captain_gameweeks: dict[int, list[int]] = {
        int(
            player_id
        ): []
        for player_id in squad["id"]
    }

    vice_captain_gameweeks: dict[
        int,
        list[int],
    ] = {
        int(
            player_id
        ): []
        for player_id in squad["id"]
    }

    for gameweek in gameweeks:
        for index in selected_indexes:
            player_id = int(
                pool.loc[
                    index,
                    "id",
                ]
            )

            if solver.value(
                starter[
                    (
                        gameweek,
                        index,
                    )
                ]
            ) == 1:
                starter_gameweeks[
                    player_id
                ].append(
                    gameweek
                )

            if solver.value(
                captain[
                    (
                        gameweek,
                        index,
                    )
                ]
            ) == 1:
                captain_gameweeks[
                    player_id
                ].append(
                    gameweek
                )

            if solver.value(
                vice_captain[
                    (
                        gameweek,
                        index,
                    )
                ]
            ) == 1:
                vice_captain_gameweeks[
                    player_id
                ].append(
                    gameweek
                )

    def format_gameweeks(
        values: list[int],
    ) -> str:
        return ",".join(
            str(
                gameweek
            )
            for gameweek
            in values
        )

    squad["projected_start_gameweeks"] = (
        squad["id"]
        .astype(int)
        .map(
            lambda player_id: format_gameweeks(
                starter_gameweeks.get(
                    player_id,
                    [],
                )
            )
        )
    )

    squad["projected_starts"] = (
        squad["id"]
        .astype(int)
        .map(
            lambda player_id: len(
                starter_gameweeks.get(
                    player_id,
                    [],
                )
            )
        )
        .astype(int)
    )

    squad["projected_captain_gameweeks"] = (
        squad["id"]
        .astype(int)
        .map(
            lambda player_id: format_gameweeks(
                captain_gameweeks.get(
                    player_id,
                    [],
                )
            )
        )
    )

    squad["captaincy_appearances"] = (
        squad["id"]
        .astype(int)
        .map(
            lambda player_id: len(
                captain_gameweeks.get(
                    player_id,
                    [],
                )
            )
        )
        .astype(int)
    )

    squad[
        "projected_vice_captain_gameweeks"
    ] = (
        squad["id"]
        .astype(int)
        .map(
            lambda player_id: format_gameweeks(
                vice_captain_gameweeks.get(
                    player_id,
                    [],
                )
            )
        )
    )

    return squad.reset_index(
        drop=True
    )


def _solve_lineup_aware_squad(
    players: pd.DataFrame,
    available_budget: float,
    selected_gameweeks: list[int] | tuple[int, ...] | None = None,
    fixed_player_ids: set[int] | None = None,
    preserve_player_ids: set[int] | None = None,
) -> SolvedSquad:
    """
    Select a legal squad and legal XI for every requested Gameweek.

    When fixed_player_ids is supplied, the solver evaluates that
    exact squad rather than recruiting a new one.
    """

    budget = _normalise_budget(
        available_budget
    )

    pool, gameweek_columns = (
        prepare_optimisation_pool(
            players=players,
            preserve_player_ids=(
                preserve_player_ids
            ),
        )
    )

    validate_position_pool(
        pool
    )

    gameweeks = _resolve_gameweeks(
        gameweek_columns=gameweek_columns,
        selected_gameweeks=(
            selected_gameweeks
        ),
    )

    fixed_ids = (
        {
            int(
                player_id
            )
            for player_id
            in fixed_player_ids
        }
        if fixed_player_ids
        is not None
        else None
    )

    if (
        fixed_ids is not None
        and len(
            fixed_ids
        )
        != SQUAD_SIZE
    ):
        raise ValueError(
            "A fixed squad must contain exactly "
            "15 unique player IDs."
        )

    if fixed_ids is not None:
        missing_fixed_ids = fixed_ids.difference(
            set(
                pool["id"]
            )
        )

        if missing_fixed_ids:
            raise ValueError(
                "The optimisation pool is missing fixed "
                "squad player IDs: "
                + ", ".join(
                    str(
                        player_id
                    )
                    for player_id
                    in sorted(
                        missing_fixed_ids
                    )
                )
            )

    model = cp_model.CpModel()

    selected = {
        index: model.new_bool_var(
            "select_player_"
            f"{int(player['id'])}"
        )
        for index, player
        in pool.iterrows()
    }

    starter = {
        (
            gameweek,
            index,
        ): model.new_bool_var(
            "starter_"
            f"gw_{gameweek}_"
            f"player_{int(pool.loc[index, 'id'])}"
        )
        for gameweek in gameweeks
        for index in pool.index
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
        for gameweek in gameweeks
        for index in pool.index
    }

    vice_captain = {
        (
            gameweek,
            index,
        ): model.new_bool_var(
            "vice_captain_"
            f"gw_{gameweek}_"
            f"player_{int(pool.loc[index, 'id'])}"
        )
        for gameweek in gameweeks
        for index in pool.index
    }

    model.add(
        sum(
            selected.values()
        )
        == SQUAD_SIZE
    )

    for (
        position,
        required_count,
    ) in POSITION_REQUIREMENTS.items():
        position_indexes = (
            pool.index[
                pool["position"]
                == position
            ]
            .tolist()
        )

        model.add(
            sum(
                selected[index]
                for index
                in position_indexes
            )
            == required_count
        )

    for club_name in sorted(
        pool[
            "team_name"
        ].unique()
    ):
        club_indexes = (
            pool.index[
                pool["team_name"]
                == club_name
            ]
            .tolist()
        )

        model.add(
            sum(
                selected[index]
                for index
                in club_indexes
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
            for index
            in pool.index
        )
        <= _budget_to_tenths(
            budget
        )
    )

    if fixed_ids is not None:
        for index in pool.index:
            player_id = int(
                pool.loc[
                    index,
                    "id",
                ]
            )

            if player_id in fixed_ids:
                model.add(
                    selected[index]
                    == 1
                )
            else:
                model.add(
                    selected[index]
                    == 0
                )

    for gameweek in gameweeks:
        model.add(
            sum(
                starter[
                    (
                        gameweek,
                        index,
                    )
                ]
                for index
                in pool.index
            )
            == STARTING_XI_SIZE
        )

        for position in (
            MIN_STARTERS_BY_POSITION
        ):
            position_indexes = (
                pool.index[
                    pool["position"]
                    == position
                ]
                .tolist()
            )

            model.add(
                sum(
                    starter[
                        (
                            gameweek,
                            index,
                        )
                    ]
                    for index
                    in position_indexes
                )
                >= MIN_STARTERS_BY_POSITION[
                    position
                ]
            )

            model.add(
                sum(
                    starter[
                        (
                            gameweek,
                            index,
                        )
                    ]
                    for index
                    in position_indexes
                )
                <= MAX_STARTERS_BY_POSITION[
                    position
                ]
            )

        model.add(
            sum(
                captain[
                    (
                        gameweek,
                        index,
                    )
                ]
                for index
                in pool.index
            )
            == 1
        )

        model.add(
            sum(
                vice_captain[
                    (
                        gameweek,
                        index,
                    )
                ]
                for index
                in pool.index
            )
            == 1
        )

        for index in pool.index:
            model.add(
                starter[
                    (
                        gameweek,
                        index,
                    )
                ]
                <= selected[index]
            )

            model.add(
                captain[
                    (
                        gameweek,
                        index,
                    )
                ]
                <= starter[
                    (
                        gameweek,
                        index,
                    )
                ]
            )

            model.add(
                vice_captain[
                    (
                        gameweek,
                        index,
                    )
                ]
                <= starter[
                    (
                        gameweek,
                        index,
                    )
                ]
            )

            model.add(
                captain[
                    (
                        gameweek,
                        index,
                    )
                ]
                + vice_captain[
                    (
                        gameweek,
                        index,
                    )
                ]
                <= 1
            )

    weekly_starting_score = sum(
        starter[
            (
                gameweek,
                index,
            )
        ]
        * int(
            pool.loc[
                index,
                (
                    f"gw_{gameweek}_"
                    "optimisation_score"
                ),
            ]
        )
        for gameweek in gameweeks
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
                (
                    f"gw_{gameweek}_"
                    "optimisation_score"
                ),
            ]
        )
        for gameweek in gameweeks
        for index in pool.index
    )

    bench_cover_score = sum(
        (
            selected[index]
            - starter[
                (
                    gameweek,
                    index,
                )
            ]
        )
        * int(
            round(
                int(
                    pool.loc[
                        index,
                        (
                            f"gw_{gameweek}_"
                            "optimisation_score"
                        ),
                    ]
                )
                * BENCH_COVER_WEIGHT
            )
        )
        for gameweek in gameweeks
        for index in pool.index
    )

    model.maximize(
        weekly_starting_score
        + captaincy_score
        + bench_cover_score
    )

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = (
        SOLVER_TIME_LIMIT_SECONDS
    )

    solver.parameters.num_search_workers = (
        SOLVER_WORKERS
    )

    status = solver.solve(
        model
    )

    valid_statuses = {
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    }

    if status not in valid_statuses:
        raise RuntimeError(
            "Project Airaola could not find a legal "
            "lineup-aware squad."
        )

    selected_indexes = [
        index
        for index
        in pool.index
        if solver.value(
            selected[index]
        )
        == 1
    ]

    squad = _build_squad_annotations(
        pool=pool,
        selected_indexes=selected_indexes,
        gameweeks=gameweeks,
        solver=solver,
        starter=starter,
        captain=captain,
        vice_captain=vice_captain,
    )

    starting_score_total = 0
    captain_score_total = 0
    bench_score_total = 0

    for gameweek in gameweeks:
        score_column = (
            f"gw_{gameweek}_"
            "optimisation_score"
        )

        for index in selected_indexes:
            player_score = int(
                pool.loc[
                    index,
                    score_column,
                ]
            )

            is_starter = solver.value(
                starter[
                    (
                        gameweek,
                        index,
                    )
                ]
            )

            is_captain = solver.value(
                captain[
                    (
                        gameweek,
                        index,
                    )
                ]
            )

            if is_starter == 1:
                starting_score_total += (
                    player_score
                )
            else:
                bench_score_total += int(
                    round(
                        player_score
                        * BENCH_COVER_WEIGHT
                    )
                )

            if is_captain == 1:
                captain_score_total += (
                    player_score
                )

    total_objective_score = (
        starting_score_total
        + captain_score_total
        + bench_score_total
    )

    squad_cost = round(
        float(
            squad[
                "price"
            ].sum()
        ),
        1,
    )

    secure_player_count = int(
        (
            squad[
                "minutes_security"
            ]
            >= SECURE_PLAYER_THRESHOLD
        ).sum()
    )

    return SolvedSquad(
        squad=squad,
        gameweeks=gameweeks,
        total_objective_points=round(
            total_objective_score
            / 100,
            2,
        ),
        starting_points=round(
            starting_score_total
            / 100,
            2,
        ),
        captain_points=round(
            captain_score_total
            / 100,
            2,
        ),
        bench_cover_points=round(
            bench_score_total
            / 100,
            2,
        ),
        squad_cost=squad_cost,
        bank_remaining=round(
            budget
            - squad_cost,
            1,
        ),
        secure_player_count=(
            secure_player_count
        ),
    )


def _squad_player_names(
    squad: pd.DataFrame,
    player_ids: set[int],
) -> tuple[str, ...]:
    """Return consistently ordered player names for an ID set."""

    if not player_ids:
        return tuple()

    selected = squad[
        squad["id"]
        .astype(int)
        .isin(
            player_ids
        )
    ].copy()

    selected = selected.sort_values(
        by=[
            "position",
            "player_name",
        ],
        ascending=[
            True,
            True,
        ],
    )

    return tuple(
        str(
            player_name
        )
        for player_name
        in selected[
            "player_name"
        ].tolist()
    )


def _build_chip_evaluation(
    chip_name: str,
    current_result: SolvedSquad,
    optimised_result: SolvedSquad,
    current_squad: pd.DataFrame,
    available_budget: float,
) -> SquadChipEvaluation:
    """Build the strategy-layer comparison for one squad chip."""

    current_ids = {
        int(
            player_id
        )
        for player_id
        in current_squad[
            "id"
        ]
    }

    optimised_ids = {
        int(
            player_id
        )
        for player_id
        in optimised_result.squad[
            "id"
        ]
    }

    incoming_ids = (
        optimised_ids
        - current_ids
    )

    outgoing_ids = (
        current_ids
        - optimised_ids
    )

    incoming_players = _squad_player_names(
        squad=optimised_result.squad,
        player_ids=incoming_ids,
    )

    outgoing_players = _squad_player_names(
        squad=current_squad,
        player_ids=outgoing_ids,
    )

    projected_gain = (
        optimised_result.total_objective_points
        - current_result.total_objective_points
    )

    current_next_gameweek_points = (
        current_result.total_objective_points
    )

    optimised_next_gameweek_points = (
        optimised_result.total_objective_points
    )

    next_gameweek_gain = (
        optimised_next_gameweek_points
        - current_next_gameweek_points
    )

    current_squad_value = round(
        float(
            current_squad[
                "price"
            ].sum()
        ),
        1,
    )

    reason = (
        "The optimiser compared both squads using the "
        "same legal formations, captaincy scoring and "
        "bench-cover weighting."
    )

    return SquadChipEvaluation(
        chip_name=chip_name,
        optimisation_succeeded=True,
        current_projected_points=round(
            current_result.total_objective_points,
            2,
        ),
        optimised_projected_points=round(
            optimised_result.total_objective_points,
            2,
        ),
        projected_gain=round(
            projected_gain,
            2,
        ),
        current_next_gameweek_points=round(
            current_next_gameweek_points,
            2,
        ),
        optimised_next_gameweek_points=round(
            optimised_next_gameweek_points,
            2,
        ),
        next_gameweek_gain=round(
            next_gameweek_gain,
            2,
        ),
        current_squad_value=(
            current_squad_value
        ),
        available_budget=round(
            float(
                available_budget
            ),
            1,
        ),
        optimised_squad_cost=(
            optimised_result.squad_cost
        ),
        bank_remaining=(
            optimised_result.bank_remaining
        ),
        secure_player_count=(
            optimised_result.secure_player_count
        ),
        changed_player_count=len(
            incoming_ids
        ),
        incoming_players=incoming_players,
        outgoing_players=outgoing_players,
        reason=reason,
    )


def optimise_initial_squad(
    players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select Airaola's initial lineup-aware £100m squad.

    The objective rewards weekly starting-XI points, captaincy
    points and a smaller amount of bench-cover value.
    """

    result = _solve_lineup_aware_squad(
        players=players,
        available_budget=(
            DEFAULT_BUDGET_LIMIT
        ),
    )

    return result.squad


def optimise_free_hit_squad(
    players: pd.DataFrame,
    current_squad: pd.DataFrame,
    available_budget: float,
    target_gameweek: int | None = None,
) -> tuple[
    pd.DataFrame,
    SquadChipEvaluation,
]:
    """
    Build and evaluate a temporary one-Gameweek Free Hit squad.

    The available budget should be the current squad's official
    selling value plus money held in the bank.
    """

    current_ids = {
        int(
            player_id
        )
        for player_id
        in current_squad[
            "id"
        ]
    }

    gameweek_columns = (
        discover_gameweek_projection_columns(
            players
        )
    )

    resolved_gameweek = (
        int(
            target_gameweek
        )
        if target_gameweek
        is not None
        else min(
            gameweek_columns
        )
    )

    optimised_result = (
        _solve_lineup_aware_squad(
            players=players,
            available_budget=(
                available_budget
            ),
            selected_gameweeks=[
                resolved_gameweek,
            ],
        )
    )

    current_result = (
        _solve_lineup_aware_squad(
            players=current_squad,
            available_budget=max(
                float(
                    current_squad[
                        "price"
                    ].sum()
                ),
                0.1,
            ),
            selected_gameweeks=[
                resolved_gameweek,
            ],
            fixed_player_ids=current_ids,
            preserve_player_ids=current_ids,
        )
    )

    evaluation = _build_chip_evaluation(
        chip_name=FREE_HIT,
        current_result=current_result,
        optimised_result=optimised_result,
        current_squad=current_squad,
        available_budget=available_budget,
    )

    return (
        optimised_result.squad,
        evaluation,
    )


def optimise_wildcard_squad(
    players: pd.DataFrame,
    current_squad: pd.DataFrame,
    available_budget: float,
    planning_gameweeks: (
        list[int]
        | tuple[int, ...]
        | None
    ) = None,
) -> tuple[
    pd.DataFrame,
    SquadChipEvaluation,
]:
    """
    Build and evaluate a permanent long-horizon Wildcard squad.

    The available budget should be the current squad's official
    selling value plus money held in the bank.
    """

    current_ids = {
        int(
            player_id
        )
        for player_id
        in current_squad[
            "id"
        ]
    }

    optimised_result = (
        _solve_lineup_aware_squad(
            players=players,
            available_budget=(
                available_budget
            ),
            selected_gameweeks=(
                planning_gameweeks
            ),
        )
    )

    current_result = (
        _solve_lineup_aware_squad(
            players=current_squad,
            available_budget=max(
                float(
                    current_squad[
                        "price"
                    ].sum()
                ),
                0.1,
            ),
            selected_gameweeks=(
                planning_gameweeks
            ),
            fixed_player_ids=current_ids,
            preserve_player_ids=current_ids,
        )
    )

    evaluation = _build_chip_evaluation(
        chip_name=WILDCARD,
        current_result=current_result,
        optimised_result=optimised_result,
        current_squad=current_squad,
        available_budget=available_budget,
    )

    return (
        optimised_result.squad,
        evaluation,
    )