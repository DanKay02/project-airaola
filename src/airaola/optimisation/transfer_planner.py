from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MAX_PLAYERS_PER_CLUB = 3
DEFAULT_BUDGET = 100.0
MINIMUM_TRANSFER_GAIN = 1.5
MINIMUM_MINUTES_SECURITY = 0.35


@dataclass(frozen=True)
class TransferRecommendation:
    """Represent Airaola's best one-transfer decision."""

    action: str
    player_out_id: int | None
    player_out_name: str | None
    player_in_id: int | None
    player_in_name: str | None
    position: str | None
    selling_price: float
    purchase_price: float
    bank_before: float
    bank_after: float
    projected_gain: float
    next_gameweek_gain: float
    recommendation_strength: str
    reason: str


def _numeric(dataframe: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a safe numeric column."""
    if column not in dataframe.columns:
        return pd.Series(default, index=dataframe.index, dtype=float)
    return pd.to_numeric(dataframe[column], errors="coerce").fillna(default)


def _validate_inputs(current_squad: pd.DataFrame, player_pool: pd.DataFrame) -> None:
    """Validate squad and player-pool inputs."""
    required_columns = {
        "id", "player_name", "team_name", "position", "price",
        "projected_points", "minutes_security", "status",
    }
    squad_missing = required_columns.difference(current_squad.columns)
    pool_missing = required_columns.difference(player_pool.columns)
    if squad_missing:
        raise ValueError(
            "Current squad is missing transfer columns: "
            + ", ".join(sorted(squad_missing))
        )
    if pool_missing:
        raise ValueError(
            "Player pool is missing transfer columns: "
            + ", ".join(sorted(pool_missing))
        )
    if len(current_squad) != 15:
        raise ValueError(
            "Transfer planning requires a complete 15-player current squad."
        )


def _recommendation_strength(projected_gain: float) -> str:
    if projected_gain >= 5.0:
        return "VERY STRONG"
    if projected_gain >= 3.0:
        return "STRONG"
    if projected_gain >= MINIMUM_TRANSFER_GAIN:
        return "MODERATE"
    return "HOLD"


def _club_counts(squad: pd.DataFrame) -> dict[str, int]:
    return squad["team_name"].value_counts().astype(int).to_dict()


def _candidate_is_club_legal(
    player_out: pd.Series,
    player_in: pd.Series,
    club_counts: dict[str, int],
) -> bool:
    outgoing_club = str(player_out["team_name"])
    incoming_club = str(player_in["team_name"])
    if outgoing_club == incoming_club:
        return True
    return int(club_counts.get(incoming_club, 0)) < MAX_PLAYERS_PER_CLUB


def _build_hold_recommendation(
    bank: float,
    best_gain: float = 0.0,
) -> TransferRecommendation:
    return TransferRecommendation(
        action="HOLD",
        player_out_id=None,
        player_out_name=None,
        player_in_id=None,
        player_in_name=None,
        position=None,
        selling_price=0.0,
        purchase_price=0.0,
        bank_before=round(bank, 1),
        bank_after=round(bank, 1),
        projected_gain=round(max(best_gain, 0.0), 2),
        next_gameweek_gain=0.0,
        recommendation_strength="HOLD",
        reason=(
            "No legal one-transfer move clears "
            f"the {MINIMUM_TRANSFER_GAIN:.1f}-point planning-horizon threshold."
        ),
    )


def recommend_single_transfer(
    current_squad: pd.DataFrame,
    player_pool: pd.DataFrame,
    budget_limit: float = DEFAULT_BUDGET,
    minimum_gain: float = MINIMUM_TRANSFER_GAIN,
) -> TransferRecommendation:
    """Recommend the best legal one-free-transfer move."""
    _validate_inputs(current_squad=current_squad, player_pool=player_pool)

    squad = current_squad.copy()
    pool = player_pool.copy()

    for dataframe in (squad, pool):
        dataframe["id"] = pd.to_numeric(dataframe["id"], errors="coerce")
        dataframe["price"] = _numeric(dataframe, "price")
        dataframe["projected_points"] = _numeric(dataframe, "projected_points")
        dataframe["minutes_security"] = _numeric(dataframe, "minutes_security")
        dataframe["next_gameweek_projected_points"] = _numeric(
            dataframe, "next_gameweek_projected_points"
        )

    squad = squad.dropna(subset=["id"]).copy()
    pool = pool.dropna(subset=["id"]).copy()
    squad["id"] = squad["id"].astype(int)
    pool["id"] = pool["id"].astype(int)

    current_cost = float(squad["price"].sum())
    bank = max(budget_limit - current_cost, 0.0)
    squad_ids = set(squad["id"])

    eligible_targets = pool[
        ~pool["id"].isin(squad_ids)
        & pool["status"].isin(["a", "d"])
        & (pool["minutes_security"] >= MINIMUM_MINUTES_SECURITY)
        & (pool["price"] > 0)
    ].copy()

    if eligible_targets.empty:
        return _build_hold_recommendation(bank=bank)

    club_counts = _club_counts(squad)
    best_move: dict[str, object] | None = None

    for _, player_out in squad.iterrows():
        available_funds = bank + float(player_out["price"])
        candidates = eligible_targets[
            (eligible_targets["position"] == player_out["position"])
            & (eligible_targets["price"] <= available_funds + 1e-9)
        ]

        for _, player_in in candidates.iterrows():
            if not _candidate_is_club_legal(player_out, player_in, club_counts):
                continue

            projected_gain = float(
                player_in["projected_points"] - player_out["projected_points"]
            )
            next_gameweek_gain = float(
                player_in["next_gameweek_projected_points"]
                - player_out["next_gameweek_projected_points"]
            )
            bank_after = available_funds - float(player_in["price"])
            candidate_key = (
                projected_gain,
                next_gameweek_gain,
                float(player_in["minutes_security"]),
                bank_after,
            )

            if best_move is None or candidate_key > best_move["candidate_key"]:
                best_move = {
                    "candidate_key": candidate_key,
                    "player_out": player_out,
                    "player_in": player_in,
                    "projected_gain": projected_gain,
                    "next_gameweek_gain": next_gameweek_gain,
                    "bank_after": bank_after,
                }

    if best_move is None:
        return _build_hold_recommendation(bank=bank)

    best_gain = float(best_move["projected_gain"])
    if best_gain < minimum_gain:
        return _build_hold_recommendation(bank=bank, best_gain=best_gain)

    player_out = best_move["player_out"]
    player_in = best_move["player_in"]
    strength = _recommendation_strength(best_gain)
    reason = (
        f"{player_in['player_name']} is projected to outscore "
        f"{player_out['player_name']} by {best_gain:.2f} points across "
        "the planning horizon while preserving squad legality."
    )

    return TransferRecommendation(
        action="TRANSFER",
        player_out_id=int(player_out["id"]),
        player_out_name=str(player_out["player_name"]),
        player_in_id=int(player_in["id"]),
        player_in_name=str(player_in["player_name"]),
        position=str(player_out["position"]),
        selling_price=round(float(player_out["price"]), 1),
        purchase_price=round(float(player_in["price"]), 1),
        bank_before=round(bank, 1),
        bank_after=round(float(best_move["bank_after"]), 1),
        projected_gain=round(best_gain, 2),
        next_gameweek_gain=round(float(best_move["next_gameweek_gain"]), 2),
        recommendation_strength=strength,
        reason=reason,
    )
