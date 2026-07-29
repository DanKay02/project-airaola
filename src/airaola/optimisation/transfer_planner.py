from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from ortools.sat.python import cp_model


SQUAD_SIZE = 15
BUDGET_LIMIT_TENTHS = 1000
MAX_PLAYERS_PER_CLUB = 3

MAX_FREE_TRANSFERS = 5
MAX_TRANSFERS_CONSIDERED = 5

POINTS_HIT_PER_EXTRA_TRANSFER = 4.0
BANKED_TRANSFER_VALUE = 0.75
MINIMUM_NET_STRATEGIC_GAIN = 1.5
MINIMUM_MINUTES_SECURITY = 0.35

POSITION_REQUIREMENTS = {
    "GKP": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}


@dataclass(frozen=True)
class TransferMove:
    """Represent one player leaving and one player joining."""

    player_out_id: int
    player_out_name: str

    player_in_id: int
    player_in_name: str

    position: str

    selling_price: float
    purchase_price: float

    projected_gain: float
    next_gameweek_gain: float


@dataclass(frozen=True)
class TransferPlan:
    """Represent Airaola's selected transfer strategy."""

    decision: str

    transfers: tuple[TransferMove, ...] = field(
        default_factory=tuple
    )

    free_transfers_before: int = 1
    transfers_used: int = 0
    free_transfers_spent: int = 0

    hit_transfers: int = 0
    hit_cost: float = 0.0

    gross_projected_gain: float = 0.0
    next_gameweek_gain: float = 0.0
    transfer_bank_cost: float = 0.0
    net_strategic_gain: float = 0.0

    best_rejected_transfer_count: int = 0
    best_rejected_gross_gain: float = 0.0
    best_rejected_hit_cost: float = 0.0
    best_rejected_bank_cost: float = 0.0
    best_rejected_net_gain: float = 0.0

    execution_threshold: float = (
        MINIMUM_NET_STRATEGIC_GAIN
    )

    bank_before: float = 0.0
    bank_after: float = 0.0

    free_transfers_next_gameweek: int = 1

    recommendation_strength: str = "HOLD"
    reason: str = ""


def _safe_numeric(
    dataframe: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """Return a numeric series without mutating the caller."""

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


def _validate_inputs(
    current_squad: pd.DataFrame,
    player_pool: pd.DataFrame,
    free_transfers_available: int,
) -> None:
    """Validate transfer-strategy inputs."""

    required_columns = {
        "id",
        "player_name",
        "team_name",
        "position",
        "price",
        "projected_points",
        "minutes_security",
        "status",
    }

    squad_missing = required_columns.difference(
        current_squad.columns
    )

    pool_missing = required_columns.difference(
        player_pool.columns
    )

    if squad_missing:
        raise ValueError(
            "Current squad is missing transfer columns: "
            + ", ".join(
                sorted(squad_missing)
            )
        )

    if pool_missing:
        raise ValueError(
            "Player pool is missing transfer columns: "
            + ", ".join(
                sorted(pool_missing)
            )
        )

    if len(current_squad) != SQUAD_SIZE:
        raise ValueError(
            "Transfer planning requires a complete "
            "15-player current squad."
        )

    if not (
        1
        <= free_transfers_available
        <= MAX_FREE_TRANSFERS
    ):
        raise ValueError(
            "Free transfers available must be between "
            f"1 and {MAX_FREE_TRANSFERS}."
        )


def _prepare_data(
    current_squad: pd.DataFrame,
    player_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare squad and player-pool data for CP-SAT."""

    squad = current_squad.copy()
    pool = player_pool.copy()

    for dataframe in (
        squad,
        pool,
    ):
        dataframe["id"] = pd.to_numeric(
            dataframe["id"],
            errors="coerce",
        )

        dataframe["price"] = _safe_numeric(
            dataframe,
            "price",
        )

        dataframe["projected_points"] = (
            _safe_numeric(
                dataframe,
                "projected_points",
            )
        )

        dataframe["minutes_security"] = (
            _safe_numeric(
                dataframe,
                "minutes_security",
            )
        )

        dataframe[
            "next_gameweek_projected_points"
        ] = _safe_numeric(
            dataframe,
            "next_gameweek_projected_points",
        )

    squad = squad.dropna(
        subset=["id"]
    ).copy()

    pool = pool.dropna(
        subset=["id"]
    ).copy()

    squad["id"] = (
        squad["id"]
        .astype(int)
    )

    pool["id"] = (
        pool["id"]
        .astype(int)
    )

    pool = pool[
        pool["position"].isin(
            POSITION_REQUIREMENTS
        )
        & pool["status"].isin(
            [
                "a",
                "d",
            ]
        )
        & (
            pool["minutes_security"]
            >= MINIMUM_MINUTES_SECURITY
        )
        & (
            pool["price"] > 0
        )
    ].copy()

    current_ids = set(
        squad["id"]
    )

    missing_current_ids = (
        current_ids.difference(
            set(
                pool["id"]
            )
        )
    )

    if missing_current_ids:
        current_rows = squad[
            squad["id"].isin(
                missing_current_ids
            )
        ].copy()

        pool = pd.concat(
            [
                pool,
                current_rows,
            ],
            ignore_index=True,
        )

    pool = (
        pool.drop_duplicates(
            subset=["id"],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    pool["price_tenths"] = (
        pool["price"]
        .mul(10)
        .round()
        .astype(int)
    )

    pool["projection_score"] = (
        pool["projected_points"]
        .mul(100)
        .round()
        .astype(int)
    )

    return (
        squad,
        pool,
    )


def _solve_exact_transfer_count(
    current_squad: pd.DataFrame,
    player_pool: pd.DataFrame,
    transfer_count: int,
) -> pd.DataFrame | None:
    """Find the best legal squad reachable with exactly N moves."""

    current_ids = set(
        current_squad["id"]
    )

    model = cp_model.CpModel()

    selected = {
        index: model.new_bool_var(
            f"select_player_{int(player['id'])}"
        )
        for index, player
        in player_pool.iterrows()
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
            player_pool.index[
                player_pool["position"]
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
        player_pool[
            "team_name"
        ].unique()
    ):
        club_indexes = (
            player_pool.index[
                player_pool["team_name"]
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
                player_pool.loc[
                    index,
                    "price_tenths",
                ]
            )
            for index
            in player_pool.index
        )
        <= BUDGET_LIMIT_TENTHS
    )

    incoming_indexes = [
        index
        for index
        in player_pool.index
        if int(
            player_pool.loc[
                index,
                "id",
            ]
        )
        not in current_ids
    ]

    model.add(
        sum(
            selected[index]
            for index
            in incoming_indexes
        )
        == transfer_count
    )

    model.maximize(
        sum(
            selected[index]
            * int(
                player_pool.loc[
                    index,
                    "projection_score",
                ]
            )
            for index
            in player_pool.index
        )
    )

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.num_search_workers = 8

    status = solver.solve(
        model
    )

    valid_statuses = {
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    }

    if status not in valid_statuses:
        return None

    selected_indexes = [
        index
        for index
        in player_pool.index
        if solver.value(
            selected[index]
        )
        == 1
    ]

    return (
        player_pool.loc[
            selected_indexes
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def _pair_transfer_moves(
    current_squad: pd.DataFrame,
    proposed_squad: pd.DataFrame,
) -> tuple[TransferMove, ...]:
    """Pair outgoing and incoming players by position."""

    current_ids = set(
        current_squad["id"]
    )

    proposed_ids = set(
        proposed_squad["id"]
    )

    outgoing = current_squad[
        ~current_squad["id"].isin(
            proposed_ids
        )
    ].copy()

    incoming = proposed_squad[
        ~proposed_squad["id"].isin(
            current_ids
        )
    ].copy()

    moves: list[TransferMove] = []

    for position in POSITION_REQUIREMENTS:
        position_out = (
            outgoing[
                outgoing["position"]
                == position
            ]
            .sort_values(
                by=[
                    "projected_points",
                    "price",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        position_in = (
            incoming[
                incoming["position"]
                == position
            ]
            .sort_values(
                by=[
                    "projected_points",
                    "minutes_security",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        if (
            len(position_out)
            != len(position_in)
        ):
            raise RuntimeError(
                "Transfer pairing failed because "
                "positional changes do not balance."
            )

        for row_index in range(
            len(position_out)
        ):
            player_out = (
                position_out.iloc[
                    row_index
                ]
            )

            player_in = (
                position_in.iloc[
                    row_index
                ]
            )

            projected_gain = float(
                player_in[
                    "projected_points"
                ]
                - player_out[
                    "projected_points"
                ]
            )

            next_gameweek_gain = float(
                player_in[
                    "next_gameweek_projected_points"
                ]
                - player_out[
                    "next_gameweek_projected_points"
                ]
            )

            moves.append(
                TransferMove(
                    player_out_id=int(
                        player_out["id"]
                    ),
                    player_out_name=str(
                        player_out[
                            "player_name"
                        ]
                    ),
                    player_in_id=int(
                        player_in["id"]
                    ),
                    player_in_name=str(
                        player_in[
                            "player_name"
                        ]
                    ),
                    position=position,
                    selling_price=round(
                        float(
                            player_out["price"]
                        ),
                        1,
                    ),
                    purchase_price=round(
                        float(
                            player_in["price"]
                        ),
                        1,
                    ),
                    projected_gain=round(
                        projected_gain,
                        2,
                    ),
                    next_gameweek_gain=round(
                        next_gameweek_gain,
                        2,
                    ),
                )
            )

    return tuple(
        moves
    )


def _free_transfers_next_gameweek(
    free_transfers_before: int,
    transfers_used: int,
) -> int:
    """Calculate the next Gameweek transfer bank."""

    remaining_free_transfers = max(
        free_transfers_before
        - transfers_used,
        0,
    )

    return min(
        remaining_free_transfers + 1,
        MAX_FREE_TRANSFERS,
    )


def _transfer_bank_cost(
    free_transfers_before: int,
    transfers_used: int,
) -> float:
    """Estimate the opportunity cost of spending saved transfers."""

    free_transfers_spent = min(
        free_transfers_before,
        transfers_used,
    )

    overflow_avoided = int(
        free_transfers_before
        == MAX_FREE_TRANSFERS
        and transfers_used > 0
    )

    strategically_costed_transfers = max(
        free_transfers_spent
        - overflow_avoided,
        0,
    )

    return (
        strategically_costed_transfers
        * BANKED_TRANSFER_VALUE
    )


def _recommendation_strength(
    net_gain: float,
) -> str:
    """Classify the strategic value of a transfer plan."""

    if net_gain >= 10.0:
        return "EXCEPTIONAL"

    if net_gain >= 6.0:
        return "VERY STRONG"

    if net_gain >= 3.0:
        return "STRONG"

    if net_gain >= MINIMUM_NET_STRATEGIC_GAIN:
        return "MODERATE"

    return "HOLD"


def _build_hold_plan(
    free_transfers_available: int,
    bank_before: float,
    best_rejected_plan: TransferPlan | None,
    minimum_net_gain: float,
) -> TransferPlan:
    """Create a transparent HOLD or ROLL recommendation."""

    next_transfer_bank = (
        _free_transfers_next_gameweek(
            free_transfers_before=(
                free_transfers_available
            ),
            transfers_used=0,
        )
    )

    if best_rejected_plan is None:
        rejected_transfer_count = 0
        rejected_gross_gain = 0.0
        rejected_hit_cost = 0.0
        rejected_bank_cost = 0.0
        rejected_net_gain = 0.0
    else:
        rejected_transfer_count = (
            best_rejected_plan.transfers_used
        )

        rejected_gross_gain = (
            best_rejected_plan.gross_projected_gain
        )

        rejected_hit_cost = (
            best_rejected_plan.hit_cost
        )

        rejected_bank_cost = (
            best_rejected_plan.transfer_bank_cost
        )

        rejected_net_gain = (
            best_rejected_plan.net_strategic_gain
        )

    if (
        free_transfers_available
        < MAX_FREE_TRANSFERS
    ):
        decision = "ROLL"

        reason = (
            "The best immediate plan produces "
            f"{rejected_net_gain:+.2f} net strategic "
            "points, below the "
            f"{minimum_net_gain:+.2f} execution "
            "threshold. Airaola therefore preserves "
            "the transfer and increases the bank."
        )
    else:
        decision = "HOLD"

        reason = (
            "The transfer bank is already full, but "
            "the best immediate plan produces only "
            f"{rejected_net_gain:+.2f} net strategic "
            "points, below the "
            f"{minimum_net_gain:+.2f} execution "
            "threshold."
        )

    return TransferPlan(
        decision=decision,
        free_transfers_before=(
            free_transfers_available
        ),
        free_transfers_next_gameweek=(
            next_transfer_bank
        ),
        bank_before=round(
            bank_before,
            1,
        ),
        bank_after=round(
            bank_before,
            1,
        ),
        gross_projected_gain=round(
            rejected_gross_gain,
            2,
        ),
        net_strategic_gain=round(
            rejected_net_gain,
            2,
        ),
        best_rejected_transfer_count=(
            rejected_transfer_count
        ),
        best_rejected_gross_gain=round(
            rejected_gross_gain,
            2,
        ),
        best_rejected_hit_cost=round(
            rejected_hit_cost,
            2,
        ),
        best_rejected_bank_cost=round(
            rejected_bank_cost,
            2,
        ),
        best_rejected_net_gain=round(
            rejected_net_gain,
            2,
        ),
        execution_threshold=round(
            minimum_net_gain,
            2,
        ),
        recommendation_strength="HOLD",
        reason=reason,
    )


def recommend_transfer_strategy(
    current_squad: pd.DataFrame,
    player_pool: pd.DataFrame,
    free_transfers_available: int = 1,
    minimum_net_gain: float = (
        MINIMUM_NET_STRATEGIC_GAIN
    ),
) -> TransferPlan:
    """
    Compare zero to five transfers and choose the best plan.

    The engine applies point-hit costs beyond the available
    transfer bank and a strategic opportunity cost for spending
    banked free transfers.
    """

    _validate_inputs(
        current_squad=current_squad,
        player_pool=player_pool,
        free_transfers_available=(
            free_transfers_available
        ),
    )

    squad, pool = _prepare_data(
        current_squad=current_squad,
        player_pool=player_pool,
    )

    current_projected_points = float(
        squad[
            "projected_points"
        ].sum()
    )

    current_next_gameweek_points = float(
        squad[
            "next_gameweek_projected_points"
        ].sum()
    )

    current_cost = float(
        squad[
            "price"
        ].sum()
    )

    bank_before = max(
        100.0 - current_cost,
        0.0,
    )

    best_plan: TransferPlan | None = None

    for transfer_count in range(
        1,
        MAX_TRANSFERS_CONSIDERED + 1,
    ):
        proposed_squad = (
            _solve_exact_transfer_count(
                current_squad=squad,
                player_pool=pool,
                transfer_count=transfer_count,
            )
        )

        if proposed_squad is None:
            continue

        proposed_projected_points = float(
            proposed_squad[
                "projected_points"
            ].sum()
        )

        proposed_next_gameweek_points = float(
            proposed_squad[
                "next_gameweek_projected_points"
            ].sum()
        )

        gross_gain = (
            proposed_projected_points
            - current_projected_points
        )

        next_gameweek_gain = (
            proposed_next_gameweek_points
            - current_next_gameweek_points
        )

        hit_transfers = max(
            transfer_count
            - free_transfers_available,
            0,
        )

        hit_cost = (
            hit_transfers
            * POINTS_HIT_PER_EXTRA_TRANSFER
        )

        transfer_bank_cost = (
            _transfer_bank_cost(
                free_transfers_before=(
                    free_transfers_available
                ),
                transfers_used=transfer_count,
            )
        )

        net_gain = (
            gross_gain
            - hit_cost
            - transfer_bank_cost
        )

        proposed_cost = float(
            proposed_squad[
                "price"
            ].sum()
        )

        bank_after = max(
            100.0 - proposed_cost,
            0.0,
        )

        moves = _pair_transfer_moves(
            current_squad=squad,
            proposed_squad=proposed_squad,
        )

        plan = TransferPlan(
            decision="EXECUTE",
            transfers=moves,
            free_transfers_before=(
                free_transfers_available
            ),
            transfers_used=transfer_count,
            free_transfers_spent=min(
                free_transfers_available,
                transfer_count,
            ),
            hit_transfers=hit_transfers,
            hit_cost=round(
                hit_cost,
                2,
            ),
            gross_projected_gain=round(
                gross_gain,
                2,
            ),
            next_gameweek_gain=round(
                next_gameweek_gain,
                2,
            ),
            transfer_bank_cost=round(
                transfer_bank_cost,
                2,
            ),
            net_strategic_gain=round(
                net_gain,
                2,
            ),
            execution_threshold=round(
                minimum_net_gain,
                2,
            ),
            bank_before=round(
                bank_before,
                1,
            ),
            bank_after=round(
                bank_after,
                1,
            ),
            free_transfers_next_gameweek=(
                _free_transfers_next_gameweek(
                    free_transfers_before=(
                        free_transfers_available
                    ),
                    transfers_used=transfer_count,
                )
            ),
            recommendation_strength=(
                _recommendation_strength(
                    net_gain
                )
            ),
            reason=(
                f"The {transfer_count}-transfer plan "
                f"adds {gross_gain:.2f} projected "
                "points before strategic costs and "
                f"{net_gain:.2f} points after them."
            ),
        )

        if best_plan is None:
            best_plan = plan
            continue

        if (
            plan.net_strategic_gain
            > best_plan.net_strategic_gain
        ):
            best_plan = plan
            continue

        if (
            plan.net_strategic_gain
            == best_plan.net_strategic_gain
            and plan.transfers_used
            < best_plan.transfers_used
        ):
            best_plan = plan

    if (
        best_plan is None
        or best_plan.net_strategic_gain
        < minimum_net_gain
    ):
        return _build_hold_plan(
            free_transfers_available=(
                free_transfers_available
            ),
            bank_before=bank_before,
            best_rejected_plan=best_plan,
            minimum_net_gain=minimum_net_gain,
        )

    return best_plan