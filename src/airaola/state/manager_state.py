from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


MAX_FREE_TRANSFERS = 5
DEFAULT_FREE_TRANSFERS = 1
DEFAULT_GAMEWEEK = 1
DEFAULT_BANK = 0.0

EXPECTED_SQUAD_SIZE = 15

DEFAULT_CHIPS = {
    "wildcard_1": True,
    "wildcard_2": True,
    "free_hit": True,
    "bench_boost": True,
    "triple_captain": True,
}


@dataclass
class SquadPlayerState:
    """Store one owned player's persistent squad data."""

    player_id: int
    purchase_price: float


@dataclass
class ManagerState:
    """Store Project Airaola's persistent season state."""

    current_gameweek: int = DEFAULT_GAMEWEEK
    free_transfers: int = DEFAULT_FREE_TRANSFERS
    bank: float = DEFAULT_BANK

    squad: list[SquadPlayerState] = field(
        default_factory=list
    )

    chips: dict[str, bool] = field(
        default_factory=lambda: DEFAULT_CHIPS.copy()
    )

    transfer_history: list[dict[str, Any]] = field(
        default_factory=list
    )

    @property
    def has_squad(self) -> bool:
        """Return whether a complete persistent squad exists."""

        return len(self.squad) == EXPECTED_SQUAD_SIZE


def create_blank_state() -> ManagerState:
    """Create a clean first-run manager state."""

    return ManagerState()


def _validate_state(
    state: ManagerState,
) -> None:
    """Validate persistent manager-state values."""

    if state.current_gameweek < 1:
        raise ValueError(
            "Manager-state current_gameweek must be at least 1."
        )

    if not (
        1
        <= state.free_transfers
        <= MAX_FREE_TRANSFERS
    ):
        raise ValueError(
            "Manager-state free_transfers must be "
            f"between 1 and {MAX_FREE_TRANSFERS}."
        )

    if state.bank < 0:
        raise ValueError(
            "Manager-state bank cannot be negative."
        )

    if len(state.squad) not in {
        0,
        EXPECTED_SQUAD_SIZE,
    }:
        raise ValueError(
            "Manager-state squad must contain either "
            "zero players for first setup or exactly "
            f"{EXPECTED_SQUAD_SIZE} players."
        )

    player_ids = [
        player.player_id
        for player in state.squad
    ]

    if len(player_ids) != len(set(player_ids)):
        raise ValueError(
            "Manager-state squad contains duplicate player IDs."
        )

    for player in state.squad:
        if player.player_id <= 0:
            raise ValueError(
                "Manager-state player IDs must be positive."
            )

        if player.purchase_price <= 0:
            raise ValueError(
                "Manager-state purchase prices must be positive."
            )


def _normalise_chips(
    chips: dict[str, Any] | None,
) -> dict[str, bool]:
    """Merge stored chip state with expected defaults."""

    normalised = DEFAULT_CHIPS.copy()

    if not chips:
        return normalised

    for chip_name in normalised:
        if chip_name in chips:
            normalised[chip_name] = bool(
                chips[chip_name]
            )

    return normalised


def _state_from_dict(
    raw_state: dict[str, Any],
) -> ManagerState:
    """Convert JSON-compatible data into ManagerState."""

    raw_squad = raw_state.get(
        "squad",
        [],
    )

    if not isinstance(raw_squad, list):
        raise ValueError(
            "Manager-state squad must be a list."
        )

    squad: list[SquadPlayerState] = []

    for player in raw_squad:
        if not isinstance(player, dict):
            raise ValueError(
                "Each manager-state squad entry "
                "must be an object."
            )

        squad.append(
            SquadPlayerState(
                player_id=int(
                    player["player_id"]
                ),
                purchase_price=float(
                    player["purchase_price"]
                ),
            )
        )

    transfer_history = raw_state.get(
        "transfer_history",
        [],
    )

    if not isinstance(
        transfer_history,
        list,
    ):
        raise ValueError(
            "Manager-state transfer_history "
            "must be a list."
        )

    state = ManagerState(
        current_gameweek=int(
            raw_state.get(
                "current_gameweek",
                DEFAULT_GAMEWEEK,
            )
        ),
        free_transfers=int(
            raw_state.get(
                "free_transfers",
                DEFAULT_FREE_TRANSFERS,
            )
        ),
        bank=float(
            raw_state.get(
                "bank",
                DEFAULT_BANK,
            )
        ),
        squad=squad,
        chips=_normalise_chips(
            raw_state.get("chips")
        ),
        transfer_history=transfer_history,
    )

    _validate_state(state)

    return state


def _state_to_dict(
    state: ManagerState,
) -> dict[str, Any]:
    """Convert ManagerState into JSON-compatible data."""

    _validate_state(state)

    return {
        "current_gameweek": int(
            state.current_gameweek
        ),
        "free_transfers": int(
            state.free_transfers
        ),
        "bank": round(
            float(state.bank),
            1,
        ),
        "squad": [
            {
                "player_id": int(
                    player.player_id
                ),
                "purchase_price": round(
                    float(
                        player.purchase_price
                    ),
                    1,
                ),
            }
            for player in state.squad
        ],
        "chips": {
            chip_name: bool(available)
            for chip_name, available
            in state.chips.items()
        },
        "transfer_history": list(
            state.transfer_history
        ),
    }


def load_manager_state(
    state_path: Path,
) -> ManagerState:
    """
    Load persistent manager state.

    A blank state file is created automatically when the
    requested path does not yet exist.
    """

    if not state_path.exists():
        state = create_blank_state()

        save_manager_state(
            state=state,
            state_path=state_path,
        )

        return state

    try:
        with state_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_state = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            "Manager-state file contains invalid JSON: "
            f"{state_path}"
        ) from error

    if not isinstance(raw_state, dict):
        raise ValueError(
            "Manager-state root must be a JSON object."
        )

    return _state_from_dict(raw_state)


def save_manager_state(
    state: ManagerState,
    state_path: Path,
) -> None:
    """Write persistent manager state safely to disk."""

    state_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_data = _state_to_dict(state)

    temporary_path = state_path.with_suffix(
        state_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            state_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    temporary_path.replace(
        state_path
    )


def initialise_squad_state(
    state: ManagerState,
    squad: pd.DataFrame,
) -> ManagerState:
    """
    Save the initial 15-player squad into persistent state.

    Current FPL prices are used as purchase prices during first
    registration.
    """

    required_columns = {
        "id",
        "price",
    }

    missing_columns = required_columns.difference(
        squad.columns
    )

    if missing_columns:
        raise ValueError(
            "Initial squad is missing state columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if len(squad) != EXPECTED_SQUAD_SIZE:
        raise ValueError(
            "Initial squad registration requires "
            f"exactly {EXPECTED_SQUAD_SIZE} players."
        )

    squad_state: list[SquadPlayerState] = []

    for _, player in squad.iterrows():
        squad_state.append(
            SquadPlayerState(
                player_id=int(
                    player["id"]
                ),
                purchase_price=round(
                    float(
                        player["price"]
                    ),
                    1,
                ),
            )
        )

    state.squad = squad_state

    state.bank = round(
        100.0
        - float(
            pd.to_numeric(
                squad["price"],
                errors="coerce",
            ).sum()
        ),
        1,
    )

    _validate_state(state)

    return state


def build_current_squad(
    state: ManagerState,
    player_pool: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct the saved squad using refreshed player data.

    Purchase prices are retained separately from current prices.
    """

    if not state.has_squad:
        raise ValueError(
            "A complete saved squad is required before "
            "reconstructing the current team."
        )

    required_columns = {
        "id",
        "player_name",
        "team_name",
        "position",
        "price",
    }

    missing_columns = required_columns.difference(
        player_pool.columns
    )

    if missing_columns:
        raise ValueError(
            "Player pool is missing persistent-state columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    purchase_prices = {
        player.player_id: player.purchase_price
        for player in state.squad
    }

    owned_player_ids = set(
        purchase_prices
    )

    pool = player_pool.copy()

    pool["id"] = pd.to_numeric(
        pool["id"],
        errors="coerce",
    )

    current_squad = pool[
        pool["id"].isin(
            owned_player_ids
        )
    ].copy()

    found_ids = set(
        current_squad["id"]
        .dropna()
        .astype(int)
    )

    missing_player_ids = (
        owned_player_ids.difference(
            found_ids
        )
    )

    if missing_player_ids:
        missing_display = ", ".join(
            str(player_id)
            for player_id in sorted(
                missing_player_ids
            )
        )

        raise ValueError(
            "Saved squad players were not found in "
            "the refreshed player pool: "
            f"{missing_display}"
        )

    current_squad["id"] = (
        current_squad["id"]
        .astype(int)
    )

    current_squad["purchase_price"] = (
        current_squad["id"]
        .map(purchase_prices)
        .astype(float)
    )

    return current_squad.reset_index(
        drop=True
    )


def _build_history_entry(
    state: ManagerState,
    transfer_plan: Any,
    moves: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create one persistent transfer-decision record."""

    return {
        "gameweek": int(
            state.current_gameweek
        ),
        "decision": str(
            transfer_plan.decision
        ),
        "free_transfers_before": int(
            transfer_plan.free_transfers_before
        ),
        "free_transfers_after": int(
            transfer_plan.free_transfers_next_gameweek
        ),
        "transfers_used": int(
            transfer_plan.transfers_used
        ),
        "hit_cost": float(
            transfer_plan.hit_cost
        ),
        "gross_projected_gain": float(
            transfer_plan.gross_projected_gain
        ),
        "net_strategic_gain": float(
            transfer_plan.net_strategic_gain
        ),
        "bank_before": float(
            transfer_plan.bank_before
        ),
        "bank_after": float(
            transfer_plan.bank_after
        ),
        "moves": moves,
    }


def apply_transfer_plan_to_state(
    state: ManagerState,
    transfer_plan: Any,
    player_pool: pd.DataFrame,
) -> ManagerState:
    """
    Apply a confirmed transfer decision to persistent state.

    EXECUTE changes the squad. ROLL and HOLD preserve the squad
    but still update the free-transfer bank and decision history.
    """

    decision = str(
        transfer_plan.decision
    ).upper()

    if decision in {
        "ROLL",
        "HOLD",
    }:
        state.free_transfers = int(
            transfer_plan.free_transfers_next_gameweek
        )

        state.bank = round(
            float(
                transfer_plan.bank_after
            ),
            1,
        )

        state.transfer_history.append(
            _build_history_entry(
                state=state,
                transfer_plan=transfer_plan,
                moves=[],
            )
        )

        _validate_state(state)

        return state

    if decision != "EXECUTE":
        raise ValueError(
            "Unknown transfer-plan decision: "
            f"{decision}"
        )

    if not state.has_squad:
        raise ValueError(
            "Transfers cannot be applied before the "
            "initial squad is registered."
        )

    pool = player_pool.copy()

    pool["id"] = pd.to_numeric(
        pool["id"],
        errors="coerce",
    )

    pool = pool.dropna(
        subset=["id"]
    ).copy()

    pool["id"] = (
        pool["id"]
        .astype(int)
    )

    player_lookup = (
        pool.set_index("id")
    )

    squad_lookup = {
        player.player_id: player
        for player in state.squad
    }

    history_moves: list[dict[str, Any]] = []

    for move in transfer_plan.transfers:
        outgoing_id = int(
            move.player_out_id
        )

        incoming_id = int(
            move.player_in_id
        )

        if outgoing_id not in squad_lookup:
            raise ValueError(
                "Outgoing player is not in the "
                f"saved squad: {outgoing_id}"
            )

        if incoming_id not in player_lookup.index:
            raise ValueError(
                "Incoming player was not found in "
                f"the player pool: {incoming_id}"
            )

        del squad_lookup[
            outgoing_id
        ]

        incoming_price = float(
            player_lookup.loc[
                incoming_id,
                "price",
            ]
        )

        squad_lookup[
            incoming_id
        ] = SquadPlayerState(
            player_id=incoming_id,
            purchase_price=round(
                incoming_price,
                1,
            ),
        )

        history_moves.append(
            {
                "player_out_id": outgoing_id,
                "player_out_name": str(
                    move.player_out_name
                ),
                "player_in_id": incoming_id,
                "player_in_name": str(
                    move.player_in_name
                ),
                "position": str(
                    move.position
                ),
                "selling_price": float(
                    move.selling_price
                ),
                "purchase_price": float(
                    move.purchase_price
                ),
                "projected_gain": float(
                    move.projected_gain
                ),
                "next_gameweek_gain": float(
                    move.next_gameweek_gain
                ),
            }
        )

    state.squad = list(
        squad_lookup.values()
    )

    state.bank = round(
        float(
            transfer_plan.bank_after
        ),
        1,
    )

    state.free_transfers = int(
        transfer_plan.free_transfers_next_gameweek
    )

    state.transfer_history.append(
        _build_history_entry(
            state=state,
            transfer_plan=transfer_plan,
            moves=history_moves,
        )
    )

    _validate_state(state)

    return state