from pathlib import Path

import pandas as pd
import yaml

from airaola.data.fetch_fpl_data import (
    run_recruitment_pipeline,
)
from airaola.models.squad_rules import validate_squad
from airaola.optimisation.sample_squad import (
    build_sample_squad,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def load_club_identity() -> dict:
    """Load Project Airaola's identity and manager philosophy."""

    config_path = (
        PROJECT_ROOT
        / "config"
        / "club_identity.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Club identity file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        identity = yaml.safe_load(file)

    if not identity:
        raise ValueError(
            "Club identity file is empty."
        )

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


def print_recruitment_summary(
    players: pd.DataFrame,
) -> None:
    """Print a summary of the downloaded player pool."""

    print()
    print("=" * 60)
    print("Recruitment Summary")
    print("=" * 60)

    print(f"Players loaded: {len(players)}")
    print(
        "Clubs represented: "
        f"{players['team_name'].nunique()}"
    )

    print()
    print("Players by position:")
    print(
        players["position"]
        .value_counts()
        .to_string()
    )


def print_squad_registration(
    squad: pd.DataFrame,
) -> None:
    """Validate and print the squad registration result."""

    result = validate_squad(squad)

    print()
    print("=" * 60)
    print("Squad Registration")
    print("=" * 60)

    display_columns = [
        "player_name",
        "team_name",
        "position",
        "price",
    ]

    position_order = {
        "GKP": 1,
        "DEF": 2,
        "MID": 3,
        "FWD": 4,
    }

    ordered_squad = squad.copy()

    ordered_squad["position_order"] = (
        ordered_squad["position"]
        .map(position_order)
    )

    ordered_squad = ordered_squad.sort_values(
        by=["position_order", "price"],
        ascending=[True, False],
    )

    print(
        ordered_squad[
            display_columns
        ].to_string(index=False)
    )

    print()
    print(
        f"Players registered: {result.player_count}"
    )
    print(
        f"Squad cost: £{result.total_cost:.1f}m"
    )

    if result.is_valid:
        print("Registration status: APPROVED")
        print("All FPL squad rules satisfied.")
        return

    print("Registration status: REJECTED")

    for error in result.errors:
        print(f"- {error}")


def main() -> None:
    """Run Project Airaola's recruitment and registration workflow."""

    try:
        identity = load_club_identity()
        print_identity(identity)

        players = run_recruitment_pipeline()
        print_recruitment_summary(players)

        sample_squad = build_sample_squad(players)
        print_squad_registration(sample_squad)

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