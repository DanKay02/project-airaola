from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests


FPL_BOOTSTRAP_URL = (
    "https://fantasy.premierleague.com/api/bootstrap-static/"
)
FPL_ELEMENT_SUMMARY_URL = (
    "https://fantasy.premierleague.com/api/element-summary/{player_id}/"
)

RAW_DATA_PATH = Path("data/raw/bootstrap_static.json")
PROCESSED_DATA_PATH = Path("data/processed/players.csv")
PRIOR_SEASON_DATA_PATH = Path(
    "data/processed/prior_season_history.csv"
)

REQUEST_HEADERS = {
    "User-Agent": "Project-Airaola/0.1.23",
    "Accept": "application/json",
}

HISTORY_FETCH_WORKERS = 6
HISTORY_FETCH_RETRIES = 3
HISTORY_RETRY_DELAY_SECONDS = 1.5


def fetch_bootstrap_data() -> dict[str, Any]:
    """Download the main Fantasy Premier League dataset."""

    response = requests.get(
        FPL_BOOTSTRAP_URL,
        timeout=30,
        headers=REQUEST_HEADERS,
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
    data: dict,
) -> pd.DataFrame:
    """Convert raw FPL player data into a clean player table."""

    players = pd.DataFrame(
        data["elements"]
    )

    teams = pd.DataFrame(
        data["teams"]
    )

    positions = pd.DataFrame(
        data["element_types"]
    )

    team_lookup = teams.set_index(
        "id"
    )["name"].to_dict()

    position_lookup = positions.set_index(
        "id"
    )["singular_name_short"].to_dict()

    players["player_name"] = (
        players["first_name"].fillna("")
        + " "
        + players["second_name"].fillna("")
    ).str.strip()

    players["team_id"] = players["team"]

    players["team_name"] = (
        players["team"]
        .map(team_lookup)
    )

    players["position"] = (
        players["element_type"]
        .map(position_lookup)
    )

    players["price"] = (
        pd.to_numeric(
            players["now_cost"],
            errors="coerce",
        )
        / 10
    )

    selected_columns = [
        "id",
        "player_name",
        "web_name",
        "first_name",
        "second_name",
        "team_id",
        "team_name",
        "position",
        "price",
        "status",
        "chance_of_playing_next_round",
        "news",
        "minutes",
        "starts",
        "total_points",
        "points_per_game",
        "form",
        "selected_by_percent",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "bps",
    ]

    missing_columns = [
        column
        for column in selected_columns
        if column not in players.columns
    ]

    for column in missing_columns:
        players[column] = None

    players = players[
        selected_columns
    ].copy()

    numeric_columns = [
        "price",
        "minutes",
        "starts",
        "total_points",
        "points_per_game",
        "form",
        "selected_by_percent",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "bps",
        "chance_of_playing_next_round",
    ]

    for column in numeric_columns:
        players[column] = pd.to_numeric(
            players[column],
            errors="coerce",
        )

    return players


def fetch_player_previous_season(
    player_id: int,
) -> dict[str, Any]:
    """
    Fetch the most recent previous-season record for one player.

    A row is returned even when the player has no previous FPL history.
    This allows the cache to remember that the player was checked.
    """

    url = FPL_ELEMENT_SUMMARY_URL.format(
        player_id=int(player_id)
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        HISTORY_FETCH_RETRIES + 1,
    ):
        try:
            response = requests.get(
                url,
                timeout=20,
                headers=REQUEST_HEADERS,
            )

            if response.status_code == 429:
                time.sleep(
                    HISTORY_RETRY_DELAY_SECONDS
                    * attempt
                )
                continue

            response.raise_for_status()

            payload = response.json()

            history_past = payload.get(
                "history_past",
                [],
            )

            if not history_past:
                return {
                    "id": int(player_id),
                    "prior_history_available": False,
                    "prior_season_name": None,
                    "prior_start_cost": None,
                    "prior_end_cost": None,
                    "prior_total_points": 0.0,
                    "prior_minutes": 0.0,
                    "prior_starts": 0.0,
                    "prior_goals_scored": 0.0,
                    "prior_assists": 0.0,
                    "prior_clean_sheets": 0.0,
                    "prior_goals_conceded": 0.0,
                    "prior_saves": 0.0,
                    "prior_bonus": 0.0,
                    "prior_bps": 0.0,
                }

            most_recent = history_past[-1]

            return {
                "id": int(player_id),
                "prior_history_available": True,
                "prior_season_name": most_recent.get(
                    "season_name"
                ),
                "prior_start_cost": _cost_to_millions(
                    most_recent.get("start_cost")
                ),
                "prior_end_cost": _cost_to_millions(
                    most_recent.get("end_cost")
                ),
                "prior_total_points": _safe_float(
                    most_recent.get("total_points")
                ),
                "prior_minutes": _safe_float(
                    most_recent.get("minutes")
                ),
                "prior_starts": _safe_float(
                    most_recent.get("starts")
                ),
                "prior_goals_scored": _safe_float(
                    most_recent.get("goals_scored")
                ),
                "prior_assists": _safe_float(
                    most_recent.get("assists")
                ),
                "prior_clean_sheets": _safe_float(
                    most_recent.get("clean_sheets")
                ),
                "prior_goals_conceded": _safe_float(
                    most_recent.get("goals_conceded")
                ),
                "prior_saves": _safe_float(
                    most_recent.get("saves")
                ),
                "prior_bonus": _safe_float(
                    most_recent.get("bonus")
                ),
                "prior_bps": _safe_float(
                    most_recent.get("bps")
                ),
            }

        except (
            requests.RequestException,
            ValueError,
            TypeError,
        ) as error:
            last_error = error

            if attempt < HISTORY_FETCH_RETRIES:
                time.sleep(
                    HISTORY_RETRY_DELAY_SECONDS
                    * attempt
                )

    raise RuntimeError(
        "Could not fetch previous-season history for "
        f"player ID {player_id}."
    ) from last_error


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert an API value to float safely."""

    try:
        if value is None:
            return default

        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _cost_to_millions(
    value: Any,
) -> float | None:
    """Convert FPL tenths-of-a-million cost to millions."""

    if value is None:
        return None

    try:
        return float(value) / 10
    except (
        TypeError,
        ValueError,
    ):
        return None


def _empty_prior_history_table() -> pd.DataFrame:
    """Return an empty table with the expected prior-history schema."""

    return pd.DataFrame(
        columns=[
            "id",
            "prior_history_available",
            "prior_season_name",
            "prior_start_cost",
            "prior_end_cost",
            "prior_total_points",
            "prior_minutes",
            "prior_starts",
            "prior_goals_scored",
            "prior_assists",
            "prior_clean_sheets",
            "prior_goals_conceded",
            "prior_saves",
            "prior_bonus",
            "prior_bps",
        ]
    )


def load_prior_season_cache() -> pd.DataFrame:
    """Load cached previous-season data if available."""

    if not PRIOR_SEASON_DATA_PATH.exists():
        return _empty_prior_history_table()

    try:
        cached = pd.read_csv(
            PRIOR_SEASON_DATA_PATH
        )
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return _empty_prior_history_table()

    if "id" not in cached.columns:
        return _empty_prior_history_table()

    cached["id"] = pd.to_numeric(
        cached["id"],
        errors="coerce",
    )

    cached = cached.dropna(
        subset=["id"]
    ).copy()

    cached["id"] = cached["id"].astype(int)

    return cached


def save_prior_season_cache(
    history: pd.DataFrame,
) -> None:
    """Save previous-season player history for reuse."""

    PRIOR_SEASON_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        PRIOR_SEASON_DATA_PATH,
        index=False,
        encoding="utf-8",
    )


def fetch_missing_prior_history(
    player_ids: list[int],
) -> pd.DataFrame:
    """Fetch prior-season records for IDs not already cached."""

    if not player_ids:
        return _empty_prior_history_table()

    print(
        "Recruitment Department: fetching previous-season "
        f"history for {len(player_ids)} players..."
    )

    rows: list[dict[str, Any]] = []
    failures: list[int] = []

    with ThreadPoolExecutor(
        max_workers=HISTORY_FETCH_WORKERS
    ) as executor:
        futures = {
            executor.submit(
                fetch_player_previous_season,
                player_id,
            ): player_id
            for player_id in player_ids
        }

        for completed_count, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            player_id = futures[future]

            try:
                rows.append(
                    future.result()
                )
            except Exception:
                failures.append(
                    int(player_id)
                )

            if (
                completed_count % 50 == 0
                or completed_count == len(player_ids)
            ):
                print(
                    "Recruitment Department: previous-season "
                    f"history {completed_count}/"
                    f"{len(player_ids)} checked."
                )

    if failures:
        print(
            "Recruitment Department warning: previous-season "
            f"history failed for {len(failures)} players. "
            "Those players will use projection fallbacks."
        )

        for player_id in failures:
            rows.append(
                {
                    "id": int(player_id),
                    "prior_history_available": False,
                    "prior_season_name": None,
                    "prior_start_cost": None,
                    "prior_end_cost": None,
                    "prior_total_points": 0.0,
                    "prior_minutes": 0.0,
                    "prior_starts": 0.0,
                    "prior_goals_scored": 0.0,
                    "prior_assists": 0.0,
                    "prior_clean_sheets": 0.0,
                    "prior_goals_conceded": 0.0,
                    "prior_saves": 0.0,
                    "prior_bonus": 0.0,
                    "prior_bps": 0.0,
                }
            )

    if not rows:
        return _empty_prior_history_table()

    return pd.DataFrame(rows)


def build_prior_season_history(
    players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a cached prior-season table for the current FPL player pool.

    Existing cached rows are reused. Only unseen player IDs are fetched.
    """

    cached = load_prior_season_cache()

    current_ids = {
        int(player_id)
        for player_id in pd.to_numeric(
            players["id"],
            errors="coerce",
        ).dropna()
    }

    cached_ids = set()

    if not cached.empty:
        cached_ids = set(
            cached["id"].astype(int)
        )

    missing_ids = sorted(
        current_ids.difference(
            cached_ids
        )
    )

    fetched = fetch_missing_prior_history(
        missing_ids
    )

    if cached.empty:
        combined = fetched.copy()
    elif fetched.empty:
        combined = cached.copy()
    else:
        combined = pd.concat(
            [
                cached,
                fetched,
            ],
            ignore_index=True,
        )

    if combined.empty:
        combined = _empty_prior_history_table()
    else:
        combined = (
            combined
            .drop_duplicates(
                subset=["id"],
                keep="last",
            )
            .sort_values("id")
            .reset_index(drop=True)
        )

    save_prior_season_cache(
        combined
    )

    return combined


def merge_prior_season_history(
    players: pd.DataFrame,
    prior_history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach previous-season data to the current player table."""

    if prior_history.empty:
        merged = players.copy()

        fallback_columns = {
            "prior_history_available": False,
            "prior_season_name": None,
            "prior_start_cost": None,
            "prior_end_cost": None,
            "prior_total_points": 0.0,
            "prior_minutes": 0.0,
            "prior_starts": 0.0,
            "prior_goals_scored": 0.0,
            "prior_assists": 0.0,
            "prior_clean_sheets": 0.0,
            "prior_goals_conceded": 0.0,
            "prior_saves": 0.0,
            "prior_bonus": 0.0,
            "prior_bps": 0.0,
        }

        for column, default in fallback_columns.items():
            merged[column] = default

        return merged

    merged = players.merge(
        prior_history,
        on="id",
        how="left",
    )

    merged["prior_history_available"] = (
        merged["prior_history_available"]
        .fillna(False)
        .astype(bool)
    )

    numeric_prior_columns = [
        "prior_start_cost",
        "prior_end_cost",
        "prior_total_points",
        "prior_minutes",
        "prior_starts",
        "prior_goals_scored",
        "prior_assists",
        "prior_clean_sheets",
        "prior_goals_conceded",
        "prior_saves",
        "prior_bonus",
        "prior_bps",
    ]

    for column in numeric_prior_columns:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        ).fillna(0.0)

    return merged


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

    players = build_player_dataframe(
        data
    )

    prior_history = build_prior_season_history(
        players
    )

    players = merge_prior_season_history(
        players=players,
        prior_history=prior_history,
    )

    save_processed_data(
        players
    )

    prior_available = int(
        players[
            "prior_history_available"
        ].sum()
    )

    print(
        "Recruitment Department: "
        f"{len(players)} players registered."
    )
    print(
        "Recruitment Department: "
        f"{prior_available} players have a usable "
        "previous-season FPL prior."
    )
    print(
        f"Raw data saved to: "
        f"{RAW_DATA_PATH}"
    )
    print(
        "Previous-season cache saved to: "
        f"{PRIOR_SEASON_DATA_PATH}"
    )
    print(
        f"Processed data saved to: "
        f"{PROCESSED_DATA_PATH}"
    )

    return players, data
