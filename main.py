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
from airaola.optimisation.lineup_selector import (
    select_gameweek_team,
)
from airaola.optimisation.squad_optimiser import (
    optimise_initial_squad,
)
from airaola.optimisation.transfer_planner import (
    TransferPlan,
    recommend_transfer_strategy,
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
        "start_security",
        "minutes_security",
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
        "start_security",
        "minutes_security",
        "fixture_count",
        "average_fixture_difficulty",
        "expected_minutes",
        "projected_points",
        "projection_value",
        "projected_start_gameweeks",
        "projected_starts",
        "projected_captain_gameweeks",
        "captaincy_appearances",
        "projected_vice_captain_gameweeks",
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


def print_starting_xi(
    starting_xi: pd.DataFrame,
) -> None:
    """Display Airaola's selected starting XI."""

    gameweek = int(
        starting_xi["next_gameweek"].iloc[0]
    )

    print()
    print("=" * 60)
    print(f"Gameweek {gameweek} Starting XI")
    print("=" * 60)

    position_order = {
        "GKP": 1,
        "DEF": 2,
        "MID": 3,
        "FWD": 4,
    }

    display = starting_xi.copy()

    display["position_order"] = (
        display["position"]
        .map(position_order)
    )

    display["role"] = ""

    display.loc[
        display["is_captain"],
        "role",
    ] = "CAPTAIN"

    display.loc[
        display["is_vice_captain"],
        "role",
    ] = "VICE-CAPTAIN"

    display = display.sort_values(
        by=[
            "position_order",
            "next_gameweek_projected_points",
        ],
        ascending=[
            True,
            False,
        ],
    )

    columns = [
        "player_name",
        "team_name",
        "position",
        "role",
        "minutes_security",
        "next_fixture_count",
        "next_gameweek_expected_minutes",
        "next_gameweek_projected_points",
    ]

    print(
        display[
            columns
        ].to_string(
            index=False,
            col_space=14,
        )
    )

    formation = (
        display["position"]
        .value_counts()
        .to_dict()
    )

    print()
    print(
        "Formation: "
        f"{formation.get('DEF', 0)}-"
        f"{formation.get('MID', 0)}-"
        f"{formation.get('FWD', 0)}"
    )

    captain = display[
        display["is_captain"]
    ].iloc[0]

    vice_captain = display[
        display["is_vice_captain"]
    ].iloc[0]

    print(
        f"Captain: {captain['player_name']}"
    )

    print(
        "Vice-captain: "
        f"{vice_captain['player_name']}"
    )


def print_bench(
    bench: pd.DataFrame,
) -> None:
    """Display Airaola's substitute order."""

    print()
    print("=" * 60)
    print("Substitutes")
    print("=" * 60)

    display = bench.sort_values(
        by="bench_order"
    )

    columns = [
        "bench_order",
        "player_name",
        "team_name",
        "position",
        "minutes_security",
        "next_fixture_count",
        "next_gameweek_expected_minutes",
        "next_gameweek_projected_points",
    ]

    print(
        display[
            columns
        ].to_string(
            index=False,
            col_space=14,
        )
    )


def print_transfer_plan(
    plan: TransferPlan,
) -> None:
    """Display Airaola's transfer-bank strategy."""

    print()
    print("=" * 60)
    print("Transfer Strategy Department")
    print("=" * 60)

    print(
        "Free transfers available: "
        f"{plan.free_transfers_before}"
    )
    print(
        "Money in bank before: "
        f"£{plan.bank_before:.1f}m"
    )

    if plan.decision in {"ROLL", "HOLD"}:
        print()
        print(f"Decision: {plan.decision} TRANSFER")
        print(
            "Recommendation strength: "
            f"{plan.recommendation_strength}"
        )
        print(
            "Projected free transfers next Gameweek: "
            f"{plan.free_transfers_next_gameweek}"
        )
        print(
            "Best immediate projected gain: "
            f"{plan.gross_projected_gain:+.2f}"
        )
        print(
            f"Reason: {plan.reason}"
        )
        return

    print()
    print(
        "Decision: EXECUTE "
        f"{plan.transfers_used} "
        "TRANSFER"
        f"{'S' if plan.transfers_used != 1 else ''}"
    )
    print(
        "Recommendation strength: "
        f"{plan.recommendation_strength}"
    )
    print(
        "Free transfers spent: "
        f"{plan.free_transfers_spent}"
    )
    print(
        "Hit transfers: "
        f"{plan.hit_transfers}"
    )
    print(
        "Hit cost: "
        f"-{plan.hit_cost:.0f} points"
    )

    print()
    print("Recommended transfers:")

    for transfer_number, move in enumerate(
        plan.transfers,
        start=1,
    ):
        print(
            f"{transfer_number}. "
            f"SELL {move.player_out_name} "
            f"£{move.selling_price:.1f}m "
            "→ "
            f"BUY {move.player_in_name} "
            f"£{move.purchase_price:.1f}m "
            f"({move.position})"
        )
        print(
            "   Five-Gameweek gain: "
            f"{move.projected_gain:+.2f} | "
            "Next-Gameweek gain: "
            f"{move.next_gameweek_gain:+.2f}"
        )

    print()
    print(
        "Gross five-Gameweek gain: "
        f"{plan.gross_projected_gain:+.2f}"
    )
    print(
        "Next-Gameweek gain: "
        f"{plan.next_gameweek_gain:+.2f}"
    )
    print(
        "Transfer-bank opportunity cost: "
        f"-{plan.transfer_bank_cost:.2f}"
    )
    print(
        "Hit cost: "
        f"-{plan.hit_cost:.2f}"
    )
    print(
        "Net strategic gain: "
        f"{plan.net_strategic_gain:+.2f}"
    )
    print(
        "Money in bank after: "
        f"£{plan.bank_after:.1f}m"
    )
    print(
        "Projected free transfers next Gameweek: "
        f"{plan.free_transfers_next_gameweek}"
    )
    print(
        f"Reason: {plan.reason}"
    )


def main() -> None:
    """Run recruitment, projections and matchday selection."""

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

        starting_xi, bench, _, _ = (
            select_gameweek_team(squad)
        )

        print_starting_xi(starting_xi)
        print_bench(bench)

        free_transfers_available = 1

        print()
        print(
            "Transfer Strategy Department: "
            "evaluating zero-to-five-transfer plans..."
        )

        transfer_plan = (
            recommend_transfer_strategy(
                current_squad=squad,
                player_pool=projected_players,
                free_transfers_available=(
                    free_transfers_available
                ),
            )
        )

        print_transfer_plan(
            transfer_plan
        )

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