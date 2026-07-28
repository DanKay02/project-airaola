from pathlib import Path

import pandas as pd
import yaml

from airaola.data.fetch_fpl_data import (
    run_recruitment_pipeline,
)
from airaola.data.fixtures import (
    run_fixture_pipeline,
)
from airaola.models.projections import (
    build_player_projections,
)
from airaola.models.squad_rules import (
    validate_squad,
)
from airaola.optimisation.squad_optimiser import (
    optimise_initial_squad,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def load_club_identity() -> dict:
    """Load Project Airaola's identity and philosophy."""

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
    """Print Project Airaola's identity."""

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
        f"{manager['planning_horizon_gameweeks']} "
        "Gameweeks"
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


def print_fixture_summary(
    gameweek_map: pd.DataFrame,
) -> None:
    """Print blanks, doubles and the fixture map."""

    print()
    print("=" * 60)
    print("Fixture Intelligence")
    print("=" * 60)

    doubles = gameweek_map[
        gameweek_map["classification"] == "DOUBLE"
    ]

    blanks = gameweek_map[
        gameweek_map["classification"] == "BLANK"
    ]

    if doubles.empty:
        print("Double Gameweeks detected: none")
    else:
        print("Double Gameweeks detected:")

        double_columns = [
            "team_name",
            "event",
            "opponents",
            "average_difficulty",
        ]

        print(
            doubles[
                double_columns
            ].to_string(
                index=False,
                col_space=12,
            )
        )

    print()

    if blanks.empty:
        print("Blank Gameweeks detected: none")
    else:
        print(
            "Blank team-Gameweeks detected: "
            f"{len(blanks)}"
        )

        blank_columns = [
            "team_name",
            "event",
            "classification",
        ]

        print(
            blanks[
                blank_columns
            ].to_string(
                index=False,
                col_space=12,
            )
        )

    print()
    print("Planning-horizon fixture map:")

    display_columns = [
        "team_name",
        "event",
        "fixture_count",
        "classification",
        "opponents",
        "average_difficulty",
    ]

    print(
        gameweek_map[
            display_columns
        ].to_string(
            index=False,
            col_space=12,
        )
    )


def print_projection_summary(
    players: pd.DataFrame,
    planning_horizon: int,
) -> None:
    """Display the highest fixture-adjusted projections."""

    print()
    print("=" * 60)
    print("Projection Department")
    print("=" * 60)

    print(
        "Projection horizon: "
        f"{planning_horizon} Gameweeks"
    )

    columns = [
        "player_name",
        "team_name",
        "position",
        "price",
        "fixture_count",
        "average_fixture_difficulty",
        "expected_minutes",
        "projected_points",
        "projection_value",
    ]

    leaders = (
        players[
            columns
        ]
        .sort_values(
            by="projected_points",
            ascending=False,
        )
        .head(10)
    )

    print()
    print("Highest projected players:")

    print(
        leaders.to_string(
            index=False,
            col_space=14,
        )
    )


def print_optimised_squad(
    squad: pd.DataFrame,
) -> None:
    """Validate and display Airaola's selected squad."""

    result = validate_squad(squad)

    print()
    print("=" * 60)
    print("Projected First Team Selection")
    print("=" * 60)

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
        by=[
            "position_order",
            "projected_points",
        ],
        ascending=[
            True,
            False,
        ],
    )

    display_columns = [
        "player_name",
        "team_name",
        "position",
        "price",
        "fixture_count",
        "average_fixture_difficulty",
        "expected_minutes",
        "projected_points",
        "projection_value",
    ]

    print(
        ordered_squad[
            display_columns
        ].to_string(
            index=False,
            col_space=14,
        )
    )

    total_projected_points = float(
        squad["projected_points"].sum()
    )

    budget_remaining = (
        100.0 - result.total_cost
    )

    print()
    print(
        f"Players selected: {result.player_count}"
    )
    print(
        f"Squad cost: £{result.total_cost:.1f}m"
    )
    print(
        "Budget remaining: "
        f"£{budget_remaining:.1f}m"
    )
    print(
        "Combined projected points: "
        f"{total_projected_points:.2f}"
    )

    if result.is_valid:
        print("Registration status: APPROVED")
        print(
            "Optimisation status: "
            "PROJECTED SQUAD FOUND"
        )
        return

    print("Registration status: REJECTED")

    for error in result.errors:
        print(f"- {error}")


def main() -> None:
    """Run recruitment, fixtures, projections and optimisation."""

    try:
        identity = load_club_identity()
        print_identity(identity)

        planning_horizon = int(
            identity["manager"][
                "planning_horizon_gameweeks"
            ]
        )

        players, bootstrap_data = (
            run_recruitment_pipeline()
        )

        print_recruitment_summary(players)

        fixture_rows, gameweek_map = (
            run_fixture_pipeline(
                bootstrap_data=bootstrap_data,
                planning_horizon=planning_horizon,
            )
        )

        print_fixture_summary(gameweek_map)

        print()
        print(
            "Projection Department: "
            "calculating future player scores..."
        )

        projected_players = (
            build_player_projections(
                players=players,
                team_fixtures=fixture_rows,
                planning_horizon=planning_horizon,
            )
        )

        print_projection_summary(
            projected_players,
            planning_horizon,
        )

        print()
        print(
            "First Team Department: "
            "optimising projected squad..."
        )

        squad = optimise_initial_squad(
            projected_players
        )

        print_optimised_squad(squad)

    except FileNotFoundError as error:
        print(
            f"Configuration error: {error}"
        )
        raise SystemExit(1) from error

    except ValueError as error:
        print(
            f"Data validation error: {error}"
        )
        raise SystemExit(1) from error

    except RuntimeError as error:
        print(
            f"Optimisation error: {error}"
        )
        raise SystemExit(1) from error

    except Exception as error:
        print(
            f"Unexpected error: {error}"
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()