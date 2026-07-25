from __future__ import annotations

import pandas as pd


POSITION_REQUIREMENTS = {
    "GKP": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}


def build_sample_squad(
    players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a basic legal-shaped squad for testing.

    This does not optimise projected points yet.
    It selects affordable players while respecting
    positional and club limits.
    """

    selected_rows: list[pd.Series] = []
    club_counts: dict[str, int] = {}

    available_players = players[
        players["status"].isin(["a", "d"])
    ].copy()

    available_players = available_players.sort_values(
        by=["price", "total_points"],
        ascending=[True, False],
    )

    for position, required_count in POSITION_REQUIREMENTS.items():
        position_pool = available_players[
            available_players["position"] == position
        ]

        selected_for_position = 0

        for _, player in position_pool.iterrows():
            club_name = player["team_name"]
            current_club_count = club_counts.get(
                club_name,
                0,
            )

            if current_club_count >= 3:
                continue

            selected_rows.append(player)

            club_counts[club_name] = (
                current_club_count + 1
            )

            selected_for_position += 1

            if selected_for_position == required_count:
                break

        if selected_for_position != required_count:
            raise ValueError(
                f"Unable to select {required_count} players "
                f"for position {position}."
            )

    return pd.DataFrame(
        selected_rows
    ).reset_index(drop=True)