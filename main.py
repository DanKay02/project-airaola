import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

from airaola.finance.price_engine import (
    SquadValueResult,
    calculate_squad_value,
)
from airaola.data.deadline_intelligence import (
    ADVANCEMENT_REQUIRED,
    MULTIPLE_ADVANCEMENTS_REQUIRED,
    SEASON_COMPLETE,
    SEASON_NOT_STARTED,
    STATE_AHEAD,
    DeadlineIntelligence,
    analyse_deadline_intelligence,
)
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
from airaola.notifications.email_delivery import (
    load_email_configuration,
    send_report_email,
)
from airaola.optimisation.lineup_selector import (
    select_gameweek_team,
)
from airaola.optimisation.squad_optimiser import (
    optimise_free_hit_squad,
    optimise_initial_squad,
    optimise_wildcard_squad,
)
from airaola.optimisation.transfer_planner import (
    TransferPlan,
    recommend_transfer_strategy,
)
from airaola.reporting.weekly_report import (
    build_weekly_report,
    save_weekly_report,
)
from airaola.strategy.chip_strategy import (
    FREE_HIT,
    WILDCARD,
    ChipRecommendation,
    SquadChipEvaluation,
    recommend_chip_strategy,
)
from airaola.state.manager_state import (
    ManagerState,
    advance_gameweek,
    apply_chip_recommendation_to_state,
    apply_transfer_plan_to_state,
    build_current_squad,
    gameweek_is_processed,
    initialise_squad_state,
    load_manager_state,
    mark_gameweek_processed,
    save_manager_state,
)


PROJECT_ROOT = Path(__file__).resolve().parent
STATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "state"
    / "manager_state.json"
)

REPORTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
)


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



def _format_gameweek(
    gameweek: int | None,
) -> str:
    """Format an optional Gameweek value for console output."""

    if gameweek is None:
        return "none"

    return str(
        gameweek
    )


def print_deadline_intelligence(
    intelligence: DeadlineIntelligence,
) -> None:
    """Display Airaola's official FPL season-clock assessment."""

    print()
    print("=" * 60)
    print("Deadline Intelligence")
    print("=" * 60)

    print(
        "Checked at: "
        f"{intelligence.checked_at_local:%Y-%m-%d %H:%M %Z}"
    )
    print(
        "Saved Gameweek: "
        f"{intelligence.saved_gameweek}"
    )
    print(
        "Official previous Gameweek: "
        + _format_gameweek(
            intelligence.official_previous_gameweek
        )
    )
    print(
        "Official current Gameweek: "
        + _format_gameweek(
            intelligence.official_current_gameweek
        )
    )
    print(
        "Official next Gameweek: "
        + _format_gameweek(
            intelligence.official_next_gameweek
        )
    )
    print(
        "Resolved active Gameweek: "
        + _format_gameweek(
            intelligence.active_gameweek
        )
    )

    print()
    print(
        "Season-clock status: "
        f"{intelligence.state_status}"
    )
    print(
        "Deadline window: "
        f"{intelligence.deadline_window}"
    )

    if (
        intelligence.next_deadline_gameweek is not None
        and intelligence.next_deadline_local is not None
    ):
        print(
            "Next deadline: "
            f"Gameweek {intelligence.next_deadline_gameweek}, "
            f"{intelligence.next_deadline_local:%Y-%m-%d %H:%M %Z}"
        )

    if intelligence.hours_until_deadline is not None:
        print(
            "Hours until deadline: "
            f"{intelligence.hours_until_deadline:.2f}"
        )

    if intelligence.advancement_count > 0:
        print(
            "Required advancement steps: "
            f"{intelligence.advancement_count}"
        )
        print(
            "Advancement target: "
            + _format_gameweek(
                intelligence.advancement_target
            )
        )
        print(
            "Single-step advancement safe: "
            + (
                "YES"
                if intelligence.safe_to_advance
                else "NO"
            )
        )

    print()
    print(
        f"Recommendation: {intelligence.recommendation}"
    )


def deadline_blocks_weekly_cycle(
    intelligence: DeadlineIntelligence,
) -> bool:
    """Return whether official season timing should stop analysis."""

    if (
        intelligence.state_status == STATE_AHEAD
        and intelligence.saved_gameweek
        == intelligence.official_next_gameweek
    ):
        return False

    return intelligence.state_status in {
        STATE_AHEAD,
        ADVANCEMENT_REQUIRED,
        MULTIPLE_ADVANCEMENTS_REQUIRED,
        SEASON_COMPLETE,
    }



def autonomous_run_is_allowed(
    arguments: argparse.Namespace,
    intelligence: DeadlineIntelligence,
) -> bool:
    """Enforce the deadline safety gate for autonomous runs."""

    if not arguments.autonomous:
        return True

    print()
    print("=" * 60)
    print("Autonomous Safety Gate")
    print("=" * 60)

    hours_remaining = intelligence.hours_until_deadline

    if hours_remaining is None:
        print("Status: BLOCKED")
        print(
            "Reason: no valid upcoming deadline could be resolved."
        )
        return False

    if hours_remaining <= 0:
        print("Status: BLOCKED")
        print(
            "Reason: the official deadline has already passed."
        )
        return False

    if hours_remaining > 24:
        print("Status: BLOCKED")
        print(
            "Reason: autonomous decisions may only be processed "
            "inside the final 24 hours before the deadline."
        )
        print(
            f"Hours remaining: {hours_remaining:.2f}"
        )
        return False

    print("Status: APPROVED")
    print(
        "The official deadline is inside the final 24-hour "
        "autonomous decision window."
    )
    print(
        f"Hours remaining: {hours_remaining:.2f}"
    )
    return True


def autonomous_sync_gameweek_state(
    arguments: argparse.Namespace,
    intelligence: DeadlineIntelligence,
    manager_state: ManagerState,
    bootstrap_data: dict,
) -> tuple[ManagerState, DeadlineIntelligence, bool]:
    """Advance one completed Gameweek when official timing requires it.

    Autonomous cloud runs may safely move the saved season clock forward
    only when the current saved Gameweek has already been fully processed
    and official FPL timing says exactly one advancement is required.
    """

    if not arguments.autonomous:
        return manager_state, intelligence, False

    if intelligence.state_status != ADVANCEMENT_REQUIRED:
        return manager_state, intelligence, False

    current_gameweek = int(manager_state.current_gameweek)

    print()
    print("=" * 60)
    print("Autonomous Season Sync")
    print("=" * 60)

    if not gameweek_is_processed(
        manager_state,
        current_gameweek,
    ):
        print("Status: BLOCKED")
        print(
            f"Gameweek {current_gameweek} is not fully processed. "
            "Airaola will not advance an incomplete lifecycle."
        )
        return manager_state, intelligence, False

    manager_state = advance_gameweek(
        state=manager_state
    )

    save_manager_state(
        state=manager_state,
        state_path=STATE_PATH,
    )

    refreshed_intelligence = analyse_deadline_intelligence(
        bootstrap_data=bootstrap_data,
        saved_gameweek=manager_state.current_gameweek,
    )

    print("Status: ADVANCED")
    print(
        f"Saved season clock advanced from Gameweek "
        f"{current_gameweek} to Gameweek "
        f"{manager_state.current_gameweek}."
    )
    print(
        "Reason: the previous Gameweek was fully processed and "
        "official FPL timing now requires the next lifecycle."
    )
    print(
        f"Manager state saved successfully: {STATE_PATH}"
    )

    return manager_state, refreshed_intelligence, True

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
    """
    Validate and display Airaola's current squad.

    Optimiser-specific planning columns are displayed when they
    are available, but are optional for squads reconstructed from
    persistent manager state.
    """

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

    core_display_columns = [
        "player_name",
        "team_name",
        "position",
        "price",
        "purchase_price",
        "start_security",
        "minutes_security",
        "fixture_count",
        "average_fixture_difficulty",
        "expected_minutes",
        "projected_points",
        "projection_value",
    ]

    optimiser_display_columns = [
        "projected_start_gameweeks",
        "projected_starts",
        "projected_captain_gameweeks",
        "captaincy_appearances",
        "projected_vice_captain_gameweeks",
    ]

    display_columns = [
        column
        for column in (
            core_display_columns
            + optimiser_display_columns
        )
        if column in ordered_squad.columns
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
        f"Squad cost: Â£{result.total_cost:.1f}m"
    )
    print(
        "Budget remaining: "
        f"Â£{budget_remaining:.1f}m"
    )
    print(
        "Combined projected points: "
        f"{total_projected_points:.2f}"
    )

    if "purchase_price" in squad.columns:
        purchase_cost = float(
            pd.to_numeric(
                squad["purchase_price"],
                errors="coerce",
            ).sum()
        )

        print(
            "Original purchase cost: "
            f"Â£{purchase_cost:.1f}m"
        )

    if result.is_valid:
        print("Registration status: APPROVED")
        print(
            "Squad status: "
            "VALID PERSISTENT SQUAD"
        )
        return

    print("Registration status: REJECTED")

    for error in result.errors:
        print(f"- {error}")


def print_squad_value(
    squad_value: SquadValueResult,
) -> None:
    """Display Airaola's official squad-value calculation."""

    print()
    print("=" * 60)
    print("Finance Department")
    print("=" * 60)

    print(
        "Original purchase value: "
        f"Â£{squad_value.purchase_value:.1f}m"
    )
    print(
        "Current market value: "
        f"Â£{squad_value.current_market_value:.1f}m"
    )
    print(
        "Official selling value: "
        f"Â£{squad_value.selling_value:.1f}m"
    )
    print(
        "Money in bank: "
        f"Â£{squad_value.bank:.1f}m"
    )
    print(
        "Total available budget: "
        f"Â£{squad_value.available_budget:.1f}m"
    )

    print()
    print(
        "Market value change: "
        f"{squad_value.market_change:+.1f}m"
    )
    print(
        "Realised squad-value change: "
        f"{squad_value.realised_change:+.1f}m"
    )
    print(
        "Total unrealised price rises: "
        f"Â£{squad_value.unrealised_profit:.1f}m"
    )
    print(
        "Profit retained if sold now: "
        f"Â£{squad_value.retained_profit:.1f}m"
    )
    print(
        "Value lost through price falls: "
        f"Â£{squad_value.lost_value:.1f}m"
    )

    player_rows = [
        {
            "player_name": result.player_name,
            "bought": result.purchase_price,
            "current": result.current_price,
            "selling": result.selling_price,
            "market_change": result.market_change,
            "realised_change": result.realised_change,
        }
        for result in squad_value.player_prices
        if (
            result.market_change != 0.0
            or result.realised_change != 0.0
        )
    ]

    if not player_rows:
        print()
        print(
            "Individual price movements: none"
        )
        return

    movements = pd.DataFrame(
        player_rows
    ).sort_values(
        by=[
            "market_change",
            "player_name",
        ],
        ascending=[
            False,
            True,
        ],
    )

    print()
    print("Individual price movements:")
    print(
        movements.to_string(
            index=False,
            col_space=12,
        )
    )


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
        f"Â£{plan.bank_before:.1f}m"
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

        print()
        print("Best rejected plan:")

        if plan.best_rejected_transfer_count == 0:
            print(
                "No legal transfer plan was available."
            )
        else:
            print(
                "Transfers considered: "
                f"{plan.best_rejected_transfer_count}"
            )
            print(
                "Gross five-Gameweek gain: "
                f"{plan.best_rejected_gross_gain:+.2f}"
            )
            print(
                "Transfer-bank opportunity cost: "
                f"-{plan.best_rejected_bank_cost:.2f}"
            )
            print(
                "Hit cost: "
                f"-{plan.best_rejected_hit_cost:.2f}"
            )
            print(
                "Net strategic gain: "
                f"{plan.best_rejected_net_gain:+.2f}"
            )
            print(
                "Execution threshold: "
                f"{plan.execution_threshold:+.2f}"
            )

        print()
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
            f"Â£{move.selling_price:.1f}m "
            "â†’ "
            f"BUY {move.player_in_name} "
            f"Â£{move.purchase_price:.1f}m "
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
        f"Â£{plan.bank_after:.1f}m"
    )
    print(
        "Projected free transfers next Gameweek: "
        f"{plan.free_transfers_next_gameweek}"
    )
    print(
        f"Reason: {plan.reason}"
    )


def print_manager_state(
    state: ManagerState,
    heading: str = "Persistent Manager State",
) -> None:
    """Display Airaola's saved season memory."""

    print()
    print("=" * 60)
    print(heading)
    print("=" * 60)
    print(
        f"Current Gameweek: {state.current_gameweek}"
    )
    print(
        "Saved squad players: "
        f"{len(state.squad)}/15"
    )
    print(
        "Free-transfer bank: "
        f"{state.free_transfers}"
    )
    print(
        f"Money in bank: Â£{state.bank:.1f}m"
    )
    print(
        "Recorded transfer decisions: "
        f"{len(state.transfer_history)}"
    )
    print(
        "Recorded chip decisions: "
        f"{len(state.chip_history)}"
    )
    print(
        "Recorded lifecycle events: "
        f"{len(state.lifecycle_history)}"
    )
    print(
        "Last processed Gameweek: "
        + (
            str(state.last_processed_gameweek)
            if state.last_processed_gameweek is not None
            else "none"
        )
    )
    print(
        "Current Gameweek status: "
        + (
            "PROCESSED"
            if state.current_gameweek_processed
            else "OPEN"
        )
    )

    active_period = (
        "first_half"
        if state.current_gameweek <= 19
        else "second_half"
    )

    active_chips = state.chips[active_period]

    print(
        "Active chip period: "
        f"{active_period.replace('_', ' ').title()}"
    )
    print(
        "Available chips: "
        + ", ".join(
            chip_name.replace("_", " ").title()
            for chip_name, available
            in active_chips.items()
            if available
        )
    )


def print_chip_squad_evaluation(
    heading: str,
    squad: pd.DataFrame,
    evaluation: SquadChipEvaluation,
) -> None:
    """Display one Free Hit or Wildcard squad comparison."""

    print()
    print("=" * 60)
    print(heading)
    print("=" * 60)

    print(
        "Optimisation status: "
        + (
            "SUCCESS"
            if evaluation.optimisation_succeeded
            else "FAILED"
        )
    )
    print(
        "Available budget: "
        f"Â£{evaluation.available_budget:.1f}m"
    )
    print(
        "Optimised squad cost: "
        f"Â£{evaluation.optimised_squad_cost:.1f}m"
    )
    print(
        "Bank remaining: "
        f"Â£{evaluation.bank_remaining:.1f}m"
    )
    print(
        "Secure players: "
        f"{evaluation.secure_player_count}/15"
    )
    print(
        "Players changed: "
        f"{evaluation.changed_player_count}"
    )
    print(
        "Long-horizon gain: "
        f"{evaluation.projected_gain:+.2f}"
    )
    print(
        "Next-Gameweek gain: "
        f"{evaluation.next_gameweek_gain:+.2f}"
    )

    print()
    print(
        "Players out: "
        + (
            ", ".join(
                evaluation.outgoing_players
            )
            if evaluation.outgoing_players
            else "none"
        )
    )
    print(
        "Players in: "
        + (
            ", ".join(
                evaluation.incoming_players
            )
            if evaluation.incoming_players
            else "none"
        )
    )

    position_order = {
        "GKP": 1,
        "DEF": 2,
        "MID": 3,
        "FWD": 4,
    }

    display = squad.copy()

    display["position_order"] = (
        display["position"]
        .map(position_order)
    )

    display = display.sort_values(
        by=[
            "position_order",
            "projected_points",
        ],
        ascending=[
            True,
            False,
        ],
    )

    columns = [
        column
        for column in [
            "player_name",
            "team_name",
            "position",
            "price",
            "minutes_security",
            "projected_points",
            "projected_start_gameweeks",
            "projected_captain_gameweeks",
        ]
        if column in display.columns
    ]

    print()
    print("Proposed squad:")
    print(
        display[
            columns
        ].to_string(
            index=False,
            col_space=14,
        )
    )


def print_chip_recommendation(
    recommendation: ChipRecommendation,
) -> None:
    """Display Airaola's chip-strategy decision."""

    print()
    print("=" * 60)
    print("Chip Strategy Department")
    print("=" * 60)

    print(
        "Gameweek: "
        f"{recommendation.current_gameweek}"
    )
    print(
        "Chip period: "
        f"{recommendation.chip_period.replace('_', ' ').title()}"
    )

    print()
    print("Candidates evaluated:")

    for candidate in recommendation.candidates:
        availability = (
            "AVAILABLE"
            if candidate.available
            else "UNAVAILABLE"
        )

        eligibility = (
            "ELIGIBLE"
            if candidate.eligible
            else "INELIGIBLE"
        )

        print()
        print(
            f"{candidate.chip_name}: "
            f"{availability} | {eligibility}"
        )
        print(
            "Projected gain: "
            f"{candidate.projected_gain:+.2f}"
        )
        print(
            "Adjusted strategic gain: "
            f"{candidate.adjusted_gain:+.2f}"
        )
        print(
            "Execution threshold: "
            f"{candidate.threshold:+.2f}"
        )
        print(
            "Strength: "
            f"{candidate.recommendation_strength}"
        )
        print(
            f"Reason: {candidate.reason}"
        )

    print()
    print("Final chip decision:")
    print(
        f"Decision: {recommendation.decision}"
    )
    print(
        "Recommendation strength: "
        f"{recommendation.recommendation_strength}"
    )
    print(
        "Projected gain: "
        f"{recommendation.projected_gain:+.2f}"
    )
    print(
        "Adjusted gain: "
        f"{recommendation.adjusted_gain:+.2f}"
    )

    if recommendation.decision != "NO CHIP":
        print(
            "Execution threshold: "
            f"{recommendation.execution_threshold:+.2f}"
        )

    if recommendation.captain_name is not None:
        print(
            "Captain considered: "
            f"{recommendation.captain_name} "
            f"({recommendation.captain_projected_points:.2f})"
        )

    print(
        "Bench projected points: "
        f"{recommendation.bench_projected_points:.2f}"
    )
    print(
        "Bench players: "
        + ", ".join(
            recommendation.bench_players
        )
    )
    print(
        f"Reason: {recommendation.reason}"
    )


def print_first_run_registration() -> None:
    """Explain the first persistent-state registration."""

    print()
    print("=" * 60)
    print("Season State Registration")
    print("=" * 60)
    print(
        "No saved squad was found."
    )
    print(
        "Airaola has prepared an optimised "
        "15-player initial squad."
    )
    print(
        "The squad will only be stored after "
        "the state change is approved."
    )


def parse_arguments() -> argparse.Namespace:
    """Parse Project Airaola's command-line run mode."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Project Airaola's weekly FPL "
            "management cycle."
        )
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--auto-apply",
        action="store_true",
        help=(
            "Apply and save Airaola's recommendation "
            "without requesting confirmation."
        ),
    )

    mode_group.add_argument(
        "--autonomous",
        action="store_true",
        help=(
            "Allow Airaola to approve and save its own "
            "weekly decisions inside the final 24 hours "
            "before the official deadline."
        ),
    )

    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Analyse and display recommendations "
            "without changing manager state."
        ),
    )

    parser.add_argument(
        "--advance-gameweek",
        action="store_true",
        help=(
            "Advance one Gameweek after the current "
            "Gameweek has been fully processed."
        ),
    )

    parser.add_argument(
        "--send-email",
        action="store_true",
        help=(
            "Send the generated weekly decision report "
            "using environment-based SMTP configuration."
        ),
    )

    arguments = parser.parse_args()

    if arguments.autonomous and arguments.advance_gameweek:
        parser.error(
            "--autonomous cannot be combined with "
            "--advance-gameweek."
        )

    return arguments


def print_run_mode(
    arguments: argparse.Namespace,
) -> None:
    """Display the active state-mutation mode."""

    print("=" * 60)

    if arguments.autonomous:
        print("Run mode: AUTONOMOUS MANAGER")
        print(
            "Airaola may approve and save its own "
            "decisions inside the final 24-hour window."
        )
    elif arguments.auto_apply:
        print("Run mode: AUTO APPLY")
        print(
            "Confirmed decisions will be saved "
            "without an interactive prompt."
        )
    elif arguments.dry_run:
        print("Run mode: DRY RUN")
        print(
            "No changes will be written to "
            "manager_state.json."
        )
    else:
        print("Run mode: INTERACTIVE")
        print(
            "State changes require manual confirmation."
        )

    print("=" * 60)
    print()


def confirm_state_change(
    arguments: argparse.Namespace,
    prompt: str,
) -> bool:
    """
    Decide whether a proposed state change may be saved.

    Auto-apply approves automatically. Dry-run always refuses.
    Interactive mode accepts only y or yes.
    """

    if arguments.autonomous:
        print()
        print(
            "Autonomous manager mode: Airaola approved "
            "its own state change."
        )
        return True

    if arguments.auto_apply:
        print()
        print(
            "Auto-apply enabled: state change approved."
        )
        return True

    if arguments.dry_run:
        print()
        print(
            "Dry-run enabled: state change not applied."
        )
        return False

    print()

    try:
        response = input(
            f"{prompt} [y/N]: "
        )
    except EOFError:
        print(
            "No interactive input was available. "
            "State change not applied."
        )
        return False

    return response.strip().lower() in {
        "y",
        "yes",
    }



def run_gameweek_advance(
    arguments: argparse.Namespace,
) -> None:
    """Advance the saved season clock by exactly one Gameweek."""

    manager_state = load_manager_state(
        STATE_PATH
    )

    print_manager_state(
        manager_state,
        heading="Loaded Manager State",
    )

    current_gameweek = int(
        manager_state.current_gameweek
    )

    if not gameweek_is_processed(
        manager_state,
        current_gameweek,
    ):
        raise ValueError(
            f"Gameweek {current_gameweek} is still open. "
            "Complete and save both weekly decisions before "
            "advancing the season."
        )

    should_advance = confirm_state_change(
        arguments=arguments,
        prompt=(
            f"Advance Project Airaola from Gameweek "
            f"{current_gameweek} to Gameweek "
            f"{current_gameweek + 1}?"
        ),
    )

    if not should_advance:
        print_manager_state(
            manager_state,
            heading="Unchanged Manager State",
        )

        print()
        print(
            "Gameweek was not advanced."
        )
        return

    manager_state = advance_gameweek(
        state=manager_state
    )

    save_manager_state(
        state=manager_state,
        state_path=STATE_PATH,
    )

    print_manager_state(
        manager_state,
        heading="Advanced Manager State",
    )

    print()
    print(
        "Gameweek advanced successfully."
    )
    print(
        "Manager state saved successfully: "
        f"{STATE_PATH}"
    )


def main() -> None:
    """Run Airaola's persistent weekly management cycle."""

    arguments = parse_arguments()

    try:
        identity = load_club_identity()
        print_identity(identity)
        print_run_mode(arguments)

        if arguments.advance_gameweek:
            run_gameweek_advance(
                arguments
            )
            return

        planning_horizon = int(
            identity["manager"][
                "planning_horizon_gameweeks"
            ]
        )

        manager_state = load_manager_state(
            STATE_PATH
        )

        print_manager_state(
            manager_state,
            heading="Loaded Manager State",
        )

        players, bootstrap_data = (
            run_recruitment_pipeline()
        )

        print_recruitment_summary(players)

        deadline_intelligence = (
            analyse_deadline_intelligence(
                bootstrap_data=bootstrap_data,
                saved_gameweek=(
                    manager_state.current_gameweek
                ),
            )
        )

        print_deadline_intelligence(
            deadline_intelligence
        )

        (
            manager_state,
            deadline_intelligence,
            autonomous_state_synced,
        ) = autonomous_sync_gameweek_state(
            arguments=arguments,
            intelligence=deadline_intelligence,
            manager_state=manager_state,
            bootstrap_data=bootstrap_data,
        )

        if autonomous_state_synced:
            print_deadline_intelligence(
                deadline_intelligence
            )

        if not autonomous_run_is_allowed(
            arguments=arguments,
            intelligence=deadline_intelligence,
        ):
            print()
            print(
                "Autonomous run stopped before optimisation, "
                "report generation, email delivery or state mutation."
            )
            return

        if deadline_blocks_weekly_cycle(
            deadline_intelligence
        ):
            print()
            print(
                "Deadline safety lock: weekly optimisation "
                "has been stopped before any state mutation."
            )

            if (
                deadline_intelligence.state_status
                == ADVANCEMENT_REQUIRED
            ):
                print(
                    "Complete the saved Gameweek lifecycle, "
                    "then run `python main.py "
                    "--advance-gameweek`."
                )
            elif (
                deadline_intelligence.state_status
                == MULTIPLE_ADVANCEMENTS_REQUIRED
            ):
                print(
                    "The saved state is several Gameweeks "
                    "behind. Advance one completed lifecycle "
                    "at a time and review the state after "
                    "each step."
                )
            elif (
                deadline_intelligence.state_status
                == STATE_AHEAD
            ):
                print(
                    "Do not advance again. Wait for official "
                    "FPL event data to catch up."
                )
            elif (
                deadline_intelligence.state_status
                == SEASON_COMPLETE
            ):
                print(
                    "The official season is complete. "
                    "No further weekly cycle is available."
                )

            return

        if (
            deadline_intelligence.state_status
            == SEASON_NOT_STARTED
        ):
            print()
            print(
                "Preseason mode: squad planning may continue. "
                "No Gameweek advancement is required."
            )

        if gameweek_is_processed(
            manager_state,
            manager_state.current_gameweek,
        ):
            print()
            print(
                "Lifecycle lock: this Gameweek has already "
                "been fully processed."
            )
            print(
                "Official timing is aligned, but the saved "
                "cycle must be advanced before another "
                "weekly analysis can begin."
            )
            print(
                "Run `python main.py --advance-gameweek`."
            )
            return

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

        if not manager_state.has_squad:
            print()
            print(
                "First Team Department: "
                "optimising initial persistent squad..."
            )

            squad = optimise_initial_squad(
                projected_players
            )

            print_first_run_registration()
            print_optimised_squad(squad)

            starting_xi, bench, _, _ = (
                select_gameweek_team(squad)
            )

            print_starting_xi(starting_xi)
            print_bench(bench)

            should_register = confirm_state_change(
                arguments=arguments,
                prompt=(
                    "Register this squad as Airaola's "
                    "persistent initial squad?"
                ),
            )

            if not should_register:
                print_manager_state(
                    manager_state,
                    heading="Unchanged Manager State",
                )

                print()
                print(
                    "Initial squad was not registered. "
                    "Manager state remains unchanged."
                )
                return

            manager_state = initialise_squad_state(
                state=manager_state,
                squad=squad,
            )

            save_manager_state(
                state=manager_state,
                state_path=STATE_PATH,
            )

            print_manager_state(
                manager_state,
                heading="Saved Manager State",
            )

            print()
            print(
                "Initial registration complete. "
                "Transfer planning begins on the "
                "next run using this saved squad."
            )
            print(
                "Manager state saved successfully: "
                f"{STATE_PATH}"
            )
            return

        squad = build_current_squad(
            state=manager_state,
            player_pool=projected_players,
        )

        print()
        print(
            "Persistent Squad Department: "
            "reconstructing saved team..."
        )

        print_optimised_squad(squad)

        squad_value = calculate_squad_value(
            squad=squad,
            bank=manager_state.bank,
        )

        print_squad_value(
            squad_value
        )

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
                    manager_state.free_transfers
                ),
                bank_available=(
                    manager_state.bank
                ),
            )
        )

        print_transfer_plan(
            transfer_plan
        )

        pre_transfer_state = copy.deepcopy(
            manager_state
        )

        should_apply = confirm_state_change(
            arguments=arguments,
            prompt=(
                "Apply this transfer decision to manager state?"
            ),
        )

        if should_apply:
            manager_state = apply_transfer_plan_to_state(
                state=manager_state,
                transfer_plan=transfer_plan,
                player_pool=projected_players,
            )

            final_squad = build_current_squad(
                state=manager_state,
                player_pool=projected_players,
            )

            if transfer_plan.decision == "EXECUTE":
                print()
                print(
                    "Persistent Squad Department: "
                    "provisional squad after transfers."
                )

                print_optimised_squad(
                    final_squad
                )

            transfer_state_message = (
                "Transfer decision approved provisionally."
            )
        else:
            final_squad = squad
            transfer_state_message = (
                "Transfer decision not applied."
            )

        starting_xi, bench, _, _ = (
            select_gameweek_team(
                final_squad
            )
        )

        print_starting_xi(starting_xi)
        print_bench(bench)

        print()
        print(
            "Chip Optimisation Department: "
            "building Free Hit and Wildcard squads..."
        )

        chip_budget = (
            squad_value.available_budget
        )

        target_gameweek = int(
            starting_xi[
                "next_gameweek"
            ].iloc[0]
        )

        free_hit_squad, free_hit_evaluation = (
            optimise_free_hit_squad(
                players=projected_players,
                current_squad=squad,
                available_budget=chip_budget,
                target_gameweek=target_gameweek,
            )
        )

        wildcard_squad, wildcard_evaluation = (
            optimise_wildcard_squad(
                players=projected_players,
                current_squad=squad,
                available_budget=chip_budget,
            )
        )

        print_chip_squad_evaluation(
            heading=(
                "Free Hit Optimisation"
            ),
            squad=free_hit_squad,
            evaluation=free_hit_evaluation,
        )

        print_chip_squad_evaluation(
            heading=(
                "Wildcard Optimisation"
            ),
            squad=wildcard_squad,
            evaluation=wildcard_evaluation,
        )

        print()
        print(
            "Chip Strategy Department: "
            "evaluating all available chips..."
        )

        chip_recommendation = recommend_chip_strategy(
            starting_xi=starting_xi,
            bench=bench,
            current_gameweek=(
                pre_transfer_state.current_gameweek
            ),
            chips=pre_transfer_state.chips,
            free_hit_evaluation=(
                free_hit_evaluation
            ),
            wildcard_evaluation=(
                wildcard_evaluation
            ),
        )

        print_chip_recommendation(
            chip_recommendation
        )

        should_apply_chip = confirm_state_change(
            arguments=arguments,
            prompt=(
                "Apply this chip decision to manager state?"
            ),
        )

        squad_chip_decisions = {
            FREE_HIT,
            WILDCARD,
        }

        selected_chip_squad = None
        selected_chip_evaluation = None

        if (
            chip_recommendation.decision
            == FREE_HIT
        ):
            selected_chip_squad = (
                free_hit_squad
            )
            selected_chip_evaluation = (
                free_hit_evaluation
            )

        if (
            chip_recommendation.decision
            == WILDCARD
        ):
            selected_chip_squad = (
                wildcard_squad
            )
            selected_chip_evaluation = (
                wildcard_evaluation
            )

        transfer_overridden = (
            should_apply_chip
            and chip_recommendation.decision
            in squad_chip_decisions
        )

        if transfer_overridden:
            # Free Hit and Wildcard supersede the ordinary transfer action,
            # so restore the pre-transfer squad, bank and free-transfer state.
            # Keep the transfer-history entry, however, because the weekly
            # lifecycle requires a transfer decision to have been recorded
            # before the Gameweek can be marked as processed.
            provisional_transfer_history = copy.deepcopy(
                manager_state.transfer_history
            )

            manager_state = copy.deepcopy(
                pre_transfer_state
            )

            manager_state.transfer_history = (
                provisional_transfer_history
            )

            if manager_state.transfer_history:
                latest_transfer_record = (
                    manager_state.transfer_history[-1]
                )

                if (
                    int(
                        latest_transfer_record.get(
                            "gameweek",
                            -1,
                        )
                    )
                    == int(
                        pre_transfer_state.current_gameweek
                    )
                ):
                    latest_transfer_record[
                        "superseded_by"
                    ] = chip_recommendation.decision

                    latest_transfer_record[
                        "applied_to_squad"
                    ] = False

            transfer_state_message = (
                "Ordinary transfer decision was recorded but "
                "superseded by "
                f"{chip_recommendation.decision}."
            )

        if should_apply_chip:
            manager_state = (
                apply_chip_recommendation_to_state(
                    state=manager_state,
                    chip_recommendation=(
                        chip_recommendation
                    ),
                    chip_squad=(
                        selected_chip_squad
                    ),
                    chip_evaluation=(
                        selected_chip_evaluation
                    ),
                )
            )

            chip_state_message = (
                "Chip decision saved successfully."
            )
        else:
            chip_state_message = (
                "Chip decision not applied."
            )

        decisions_recorded = (
            should_apply
            and should_apply_chip
        )

        lifecycle_completed = False

        if decisions_recorded:
            manager_state = mark_gameweek_processed(
                state=manager_state
            )

            lifecycle_completed = True

            save_manager_state(
                state=manager_state,
                state_path=STATE_PATH,
            )

        state_changed = (
            should_apply
            or should_apply_chip
        )

        if lifecycle_completed:
            lifecycle_status = "PROCESSED"
        elif state_changed:
            lifecycle_status = "PARTIAL"
        else:
            lifecycle_status = "OPEN"

        weekly_report = build_weekly_report(
            deadline_intelligence=(
                deadline_intelligence
            ),
            squad_value=squad_value,
            transfer_plan=transfer_plan,
            starting_xi=starting_xi,
            bench=bench,
            chip_recommendation=(
                chip_recommendation
            ),
            lifecycle_status=(
                lifecycle_status
            ),
        )

        saved_report = save_weekly_report(
            report=weekly_report,
            reports_path=REPORTS_PATH,
        )

        print()
        print("=" * 60)
        print("Weekly Decision Report")
        print("=" * 60)
        print(
            "Text report: "
            f"{saved_report.text_path}"
        )
        print(
            "HTML report: "
            f"{saved_report.html_path}"
        )
        print(
            "Report lifecycle status: "
            f"{lifecycle_status}"
        )

        if arguments.dry_run:
            print(
                "Dry-run note: report files were saved, "
                "but manager state was not changed."
            )

        if arguments.send_email:
            print()
            print("=" * 60)
            print("Email Delivery")
            print("=" * 60)

            try:
                email_configuration = (
                    load_email_configuration()
                )
            except ValueError as error:
                print("Delivery status: NOT SENT")
                print(
                    "Configuration error: "
                    f"{error}"
                )

                if arguments.autonomous:
                    raise RuntimeError(
                        "Autonomous email delivery could not be "
                        "configured. Persistent cloud state must "
                        "not be committed."
                    ) from error
            else:
                delivery_result = send_report_email(
                    report=saved_report,
                    configuration=(
                        email_configuration
                    ),
                    attach_files=True,
                )

                if delivery_result.sent:
                    print("Delivery status: SENT")
                    print(
                        "Recipient: "
                        f"{delivery_result.recipient_email}"
                    )
                    print(
                        "Subject: "
                        f"{delivery_result.subject}"
                    )
                else:
                    print("Delivery status: NOT SENT")
                    print(
                        "Recipient: "
                        f"{delivery_result.recipient_email}"
                    )
                    print(
                        "Reason: "
                        f"{delivery_result.error_message}"
                    )

                    if arguments.autonomous:
                        raise RuntimeError(
                            "Autonomous email delivery failed. "
                            "Persistent cloud state must not be "
                            "committed."
                        )
        else:
            print()
            print(
                "Email delivery: skipped. "
                "Use `--send-email` to send this report."
            )

        print_manager_state(
            manager_state,
            heading=(
                "Processed Manager State"
                if lifecycle_completed
                else (
                    "Updated Manager State"
                    if state_changed
                    else "Unchanged Manager State"
                )
            ),
        )

        print()
        print(transfer_state_message)
        print(chip_state_message)

        if lifecycle_completed:
            print(
                "Gameweek lifecycle marked as processed."
            )
            print(
                "Run `python main.py --advance-gameweek` "
                "when the next Gameweek is ready."
            )
            print(
                "Manager state saved successfully: "
                f"{STATE_PATH}"
            )
        elif state_changed:
            print(
                "The weekly lifecycle remains open because "
                "both decisions were not approved."
            )
            print(
                "Partial state changes were kept in memory "
                "only and were not written to disk."
            )
        else:
            print(
                "Manager state was left unchanged."
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

    except KeyboardInterrupt:
        print()
        print(
            "Run cancelled. Manager state was not "
            "changed after the interruption."
        )
        raise SystemExit(130)

    except Exception as error:
        print(
            f"Unexpected error: {error}"
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
