from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests


FPL_BOOTSTRAP_URL = (
    "https://fantasy.premierleague.com/api/bootstrap-static/"
)

RAW_DATA_PATH = Path("data/raw/bootstrap_static.json")
PROCESSED_DATA_PATH = Path("data/processed/players.csv")


def fetch_bootstrap_data() -> dict[str, Any]:
    """Download the main Fantasy Premier League dataset."""

    response = requests.get(
        FPL_BOOTSTRAP_URL,
        timeout=30,
        headers={
            "User-Agent": "Project-Airaola/0.1.5",
            "Accept": "application/json",
        },
    )

    response.raise_for_status()

    data = response.json()

    if "elements" not in data:
        raise ValueError(
            "FPL response does not contain the expected "
            "'elements' player data."
        )

    return data


def save_raw_data(data: dict[str, Any]) -> None:
    """Save the complete API response as JSON."""

    RAW_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RAW_DATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def build_player_dataframe(
    data: dict[str, Any],
) -> pd.DataFrame:
    """Convert raw FPL player data into a clean table."""

    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    positions = pd.DataFrame(data["element_types"])

    team_lookup = (
        teams
        .set_index("id")["name"]
        .to_dict()
    )

    position_lookup = (
        positions
        .set_index("id")["singular_name_short"]
        .to_dict()
    )

    players["team_name"] = players["team"].map(
        team_lookup
    )

    players["position"] = players[
        "element_type"
    ].map(position_lookup)

    players["player_name"] = (
        players["first_name"].str.strip()
        + " "
        + players["second_name"].str.strip()
    )

    players["price"] = (
        players["now_cost"] / 10
    )

    selected_columns = [
        "id",
        "player_name",
        "web_name",
        "team_name",
        "position",
        "price",
        "total_points",
        "minutes",
        "starts",
        "goals_scored",
        "assists",
        "clean_sheets",
        "bonus",
        "form",
        "points_per_game",
        "selected_by_percent",
        "status",
        "chance_of_playing_next_round",
    ]

    missing_columns = [
        column
        for column in selected_columns
        if column not in players.columns
    ]

    if missing_columns:
        raise ValueError(
            "FPL player data is missing expected columns: "
            + ", ".join(missing_columns)
        )

    return players[
        selected_columns
    ].copy()


def save_processed_data(
    players: pd.DataFrame,
) -> None:
    """Save cleaned player data as CSV."""

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    players.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
        encoding="utf-8",
    )


def run_recruitment_pipeline(
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the complete recruitment data pipeline."""

    print(
        "Recruitment Department: "
        "contacting FPL data source..."
    )

    data = fetch_bootstrap_data()
    save_raw_data(data)

    players = build_player_dataframe(data)
    save_processed_data(players)

    print(
        f"Recruitment Department: "
        f"{len(players)} players registered."
    )
    print(f"Raw data saved to: {RAW_DATA_PATH}")
    print(
        f"Processed data saved to: "
        f"{PROCESSED_DATA_PATH}"
    )

    return players, data
    data = fetch_bootstrap_data()
    save_raw_data(data)

    players = build_player_dataframe(data)
    save_processed_data(players)

    print(
        f"Recruitment Department: "
        f"{len(players)} players registered."
    )

    print(
        f"Raw data saved to: "
        f"{RAW_DATA_PATH}"
    )

    print(
        f"Processed data saved to: "
        f"{PROCESSED_DATA_PATH}"
    )

    return players, data