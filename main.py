from pathlib import Path

import pandas as pd
import yaml

from airaola.data.fetch_fpl_data import run_recruitment_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def load_club_identity() -> dict:
    """Load Project Airaola's identity and manager philosophy."""

    config_path = PROJECT_ROOT / "config" / "club_identity.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Club identity file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        identity = yaml.safe_load(file)

    if not identity:
        raise ValueError("Club identity file is empty.")

    return identity


def print_identity(identity: dict) -> None:
    """Print the project's identity and current version."""

    project = identity["project"]
    manager = identity["manager"]

    print("=" * 60)
    print(project["name"])
    print(project["tagline"])
    print("=" * 60)
    print(f"Version: {project['version']}")
    print(f"Manager: {manager['name']}")
    print(f"Objective: {manager['objective']}")
    print(
        "Planning horizon: "
        f"{manager['planning_horizon_gameweeks']} Gameweeks"
    )
    print()


def print_recruitment_summary(players: pd.DataFrame) -> None:
    """Print a summary of the downloaded FPL player pool."""

    print()
    print("=" * 60)
    print("Recruitment Summary")
    print("=" * 60)

    print(f"Players loaded: {len(players)}")
    print(f"Clubs represented: {players['team_name'].nunique()}")

    print()
    print("Players by position:")
    print(players["position"].value_counts().to_string())

    print()
    print("Five most expensive players:")

    expensive_players = (
        players[
            [
                "player_name",
                "team_name",
                "position",
                "price",
            ]
        ]
        .sort_values("price", ascending=False)
        .head(5)
    )

    print(expensive_players.to_string(index=False))


def main() -> None:
    """Run Project Airaola's recruitment workflow."""

    try:
        identity = load_club_identity()
        print_identity(identity)

        players = run_recruitment_pipeline()
        print_recruitment_summary(players)

    except FileNotFoundError as error:
        print(f"Configuration error: {error}")
        raise SystemExit(1) from error

    except ValueError as error:
        print(f"Data validation error: {error}")
        raise SystemExit(1) from error

    except Exception as error:
        print(f"Unexpected error: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()