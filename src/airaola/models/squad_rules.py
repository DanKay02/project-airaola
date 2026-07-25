from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


SQUAD_SIZE = 15
BUDGET_LIMIT = 100.0
MAX_PLAYERS_PER_CLUB = 3

POSITION_LIMITS = {
    "GKP": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}


@dataclass
class SquadValidationResult:
    """Store the result of an FPL squad validation."""

    is_valid: bool
    total_cost: float
    player_count: int
    errors: list[str] = field(default_factory=list)


def validate_required_columns(squad: pd.DataFrame) -> list[str]:
    """Check that the squad contains the columns required for validation."""

    required_columns = {
        "id",
        "player_name",
        "team_name",
        "position",
        "price",
    }

    missing_columns = required_columns.difference(squad.columns)

    if not missing_columns:
        return []

    return [
        "Squad data is missing required columns: "
        + ", ".join(sorted(missing_columns))
    ]


def validate_squad_size(squad: pd.DataFrame) -> list[str]:
    """Check that the squad contains exactly 15 players."""

    player_count = len(squad)

    if player_count == SQUAD_SIZE:
        return []

    return [
        f"Squad must contain exactly {SQUAD_SIZE} players. "
        f"Current total: {player_count}."
    ]


def validate_unique_players(squad: pd.DataFrame) -> list[str]:
    """Check that no player has been selected more than once."""

    duplicate_players = squad[
        squad["id"].duplicated(keep=False)
    ]

    if duplicate_players.empty:
        return []

    names = sorted(
        duplicate_players["player_name"].unique()
    )

    return [
        "Squad contains duplicate players: "
        + ", ".join(names)
    ]


def validate_positions(squad: pd.DataFrame) -> list[str]:
    """Check that the squad has the correct positional composition."""

    errors: list[str] = []

    position_counts = (
        squad["position"]
        .value_counts()
        .to_dict()
    )

    for position, required_count in POSITION_LIMITS.items():
        actual_count = int(
            position_counts.get(position, 0)
        )

        if actual_count != required_count:
            errors.append(
                f"{position} count must be {required_count}. "
                f"Current total: {actual_count}."
            )

    unexpected_positions = set(
        position_counts
    ).difference(POSITION_LIMITS)

    if unexpected_positions:
        errors.append(
            "Squad contains unexpected positions: "
            + ", ".join(sorted(unexpected_positions))
        )

    return errors


def validate_budget(squad: pd.DataFrame) -> list[str]:
    """Check that the squad does not exceed the FPL budget."""

    total_cost = float(squad["price"].sum())

    if total_cost <= BUDGET_LIMIT:
        return []

    return [
        f"Squad exceeds the £{BUDGET_LIMIT:.1f}m budget. "
        f"Current cost: £{total_cost:.1f}m."
    ]


def validate_club_limit(squad: pd.DataFrame) -> list[str]:
    """Check that no club supplies more than three players."""

    errors: list[str] = []

    club_counts = squad["team_name"].value_counts()

    invalid_clubs = club_counts[
        club_counts > MAX_PLAYERS_PER_CLUB
    ]

    for club_name, player_count in invalid_clubs.items():
        errors.append(
            f"{club_name} has {player_count} selected players. "
            f"Maximum allowed: {MAX_PLAYERS_PER_CLUB}."
        )

    return errors


def validate_squad(
    squad: pd.DataFrame,
) -> SquadValidationResult:
    """Validate a complete FPL squad against registration rules."""

    player_count = len(squad)

    required_column_errors = validate_required_columns(
        squad
    )

    if required_column_errors:
        return SquadValidationResult(
            is_valid=False,
            total_cost=0.0,
            player_count=player_count,
            errors=required_column_errors,
        )

    errors: list[str] = []

    errors.extend(validate_squad_size(squad))
    errors.extend(validate_unique_players(squad))
    errors.extend(validate_positions(squad))
    errors.extend(validate_budget(squad))
    errors.extend(validate_club_limit(squad))

    total_cost = float(squad["price"].sum())

    return SquadValidationResult(
        is_valid=not errors,
        total_cost=total_cost,
        player_count=player_count,
        errors=errors,
    )