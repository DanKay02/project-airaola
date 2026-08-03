from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from airaola.data.deadline_intelligence import (
    DeadlineIntelligence,
)
from airaola.finance.price_engine import (
    SquadValueResult,
)
from airaola.optimisation.transfer_planner import (
    TransferPlan,
)
from airaola.strategy.chip_strategy import (
    ChipRecommendation,
)


REPORTS_PATH = Path(
    "data/reports"
)


@dataclass(frozen=True)
class WeeklyReport:
    """Store one complete weekly management report."""

    gameweek: int

    text_content: str
    html_content: str

    text_path: Path | None = None
    html_path: Path | None = None


def _format_money(
    value: float,
) -> str:
    """Format an FPL monetary value."""

    return f"£{float(value):.1f}m"


def _format_points(
    value: float,
    include_sign: bool = False,
) -> str:
    """Format projected FPL points."""

    if include_sign:
        return f"{float(value):+.2f}"

    return f"{float(value):.2f}"


def _normalise_text(
    value: Any,
) -> str:
    """Convert optional report data into readable text."""

    if value is None:
        return "none"

    text = str(
        value
    ).strip()

    return text or "none"


def _extract_gameweek(
    starting_xi: pd.DataFrame,
) -> int:
    """Extract the target Gameweek from the selected XI."""

    if starting_xi.empty:
        raise ValueError(
            "Weekly report requires a non-empty starting XI."
        )

    if "next_gameweek" not in starting_xi.columns:
        raise ValueError(
            "Starting XI is missing next_gameweek."
        )

    gameweeks = (
        pd.to_numeric(
            starting_xi["next_gameweek"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    if len(gameweeks) != 1:
        raise ValueError(
            "Starting XI must contain exactly one target Gameweek."
        )

    return int(
        gameweeks[0]
    )


def _validate_starting_xi(
    starting_xi: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalise the selected starting XI."""

    required_columns = {
        "id",
        "player_name",
        "team_name",
        "position",
        "next_gameweek",
        "next_gameweek_projected_points",
        "is_captain",
        "is_vice_captain",
    }

    missing_columns = required_columns.difference(
        starting_xi.columns
    )

    if missing_columns:
        raise ValueError(
            "Starting XI is missing report columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    if len(starting_xi) != 11:
        raise ValueError(
            "Weekly report requires exactly 11 starters."
        )

    lineup = starting_xi.copy()

    lineup[
        "next_gameweek_projected_points"
    ] = pd.to_numeric(
        lineup[
            "next_gameweek_projected_points"
        ],
        errors="coerce",
    ).fillna(0.0)

    lineup["is_captain"] = (
        lineup["is_captain"]
        .astype(bool)
    )

    lineup["is_vice_captain"] = (
        lineup["is_vice_captain"]
        .astype(bool)
    )

    if int(
        lineup["is_captain"].sum()
    ) != 1:
        raise ValueError(
            "Starting XI must contain exactly one captain."
        )

    if int(
        lineup["is_vice_captain"].sum()
    ) != 1:
        raise ValueError(
            "Starting XI must contain exactly one vice-captain."
        )

    return lineup.reset_index(
        drop=True
    )


def _validate_bench(
    bench: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalise the selected bench."""

    required_columns = {
        "id",
        "player_name",
        "team_name",
        "position",
        "bench_order",
        "next_gameweek_projected_points",
    }

    missing_columns = required_columns.difference(
        bench.columns
    )

    if missing_columns:
        raise ValueError(
            "Bench is missing report columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    if len(bench) != 4:
        raise ValueError(
            "Weekly report requires exactly four substitutes."
        )

    substitutes = bench.copy()

    substitutes["bench_order"] = pd.to_numeric(
        substitutes["bench_order"],
        errors="coerce",
    )

    substitutes[
        "next_gameweek_projected_points"
    ] = pd.to_numeric(
        substitutes[
            "next_gameweek_projected_points"
        ],
        errors="coerce",
    ).fillna(0.0)

    if substitutes["bench_order"].isna().any():
        raise ValueError(
            "Bench contains invalid substitute ordering."
        )

    substitutes["bench_order"] = (
        substitutes["bench_order"]
        .astype(int)
    )

    if set(
        substitutes["bench_order"]
    ) != {
        1,
        2,
        3,
        4,
    }:
        raise ValueError(
            "Bench order must contain positions 1 to 4."
        )

    return substitutes.sort_values(
        by="bench_order"
    ).reset_index(
        drop=True
    )


def _formation(
    starting_xi: pd.DataFrame,
) -> str:
    """Return the selected outfield formation."""

    counts = (
        starting_xi["position"]
        .value_counts()
        .to_dict()
    )

    return (
        f"{counts.get('DEF', 0)}-"
        f"{counts.get('MID', 0)}-"
        f"{counts.get('FWD', 0)}"
    )


def _captain_row(
    starting_xi: pd.DataFrame,
) -> pd.Series:
    """Return the selected captain."""

    return starting_xi[
        starting_xi["is_captain"]
    ].iloc[0]


def _vice_captain_row(
    starting_xi: pd.DataFrame,
) -> pd.Series:
    """Return the selected vice-captain."""

    return starting_xi[
        starting_xi["is_vice_captain"]
    ].iloc[0]


def _transfer_actions(
    transfer_plan: TransferPlan,
) -> list[str]:
    """Build concise transfer instructions."""

    decision = str(
        transfer_plan.decision
    ).upper()

    if decision == "ROLL":
        return [
            "Roll the free transfer.",
            (
                "Expected free transfers next Gameweek: "
                f"{transfer_plan.free_transfers_next_gameweek}"
            ),
        ]

    if decision == "HOLD":
        return [
            "Make no transfer.",
            (
                "Free transfers next Gameweek: "
                f"{transfer_plan.free_transfers_next_gameweek}"
            ),
        ]

    if decision != "EXECUTE":
        return [
            f"Review unknown transfer decision: {decision}"
        ]

    actions: list[str] = []

    for move in transfer_plan.transfers:
        actions.append(
            "Sell "
            f"{move.player_out_name} "
            f"({_format_money(move.selling_price)}) "
            "and buy "
            f"{move.player_in_name} "
            f"({_format_money(move.purchase_price)})."
        )

    if transfer_plan.hit_cost > 0:
        actions.append(
            "This plan includes a "
            f"{int(transfer_plan.hit_cost)}-point hit."
        )

    return actions


def _chip_action(
    chip_recommendation: ChipRecommendation,
) -> str:
    """Build the manual chip instruction."""

    decision = str(
        chip_recommendation.decision
    ).strip().upper()

    if decision == "NO CHIP":
        return "Do not activate a chip."

    return (
        f"Activate {decision.replace('_', ' ').title()}."
    )


def _deadline_text(
    intelligence: DeadlineIntelligence,
) -> str:
    """Format the next official deadline."""

    if (
        intelligence.next_deadline_gameweek is None
        or intelligence.next_deadline_local is None
    ):
        return "No upcoming official deadline is available."

    return (
        f"Gameweek "
        f"{intelligence.next_deadline_gameweek}, "
        f"{intelligence.next_deadline_local:%Y-%m-%d %H:%M %Z}"
    )


def _lineup_rows_text(
    starting_xi: pd.DataFrame,
) -> list[str]:
    """Build plain-text starting-XI rows."""

    position_order = {
        "GKP": 1,
        "DEF": 2,
        "MID": 3,
        "FWD": 4,
    }

    lineup = starting_xi.copy()

    lineup["position_order"] = (
        lineup["position"]
        .map(position_order)
        .fillna(99)
    )

    lineup = lineup.sort_values(
        by=[
            "position_order",
            "next_gameweek_projected_points",
        ],
        ascending=[
            True,
            False,
        ],
    )

    rows: list[str] = []

    for _, player in lineup.iterrows():
        role = ""

        if bool(
            player["is_captain"]
        ):
            role = " [C]"
        elif bool(
            player["is_vice_captain"]
        ):
            role = " [VC]"

        rows.append(
            f"- {player['position']}: "
            f"{player['player_name']} "
            f"({player['team_name']})"
            f"{role} | "
            f"{_format_points(player['next_gameweek_projected_points'])} pts"
        )

    return rows


def _bench_rows_text(
    bench: pd.DataFrame,
) -> list[str]:
    """Build plain-text bench rows."""

    rows: list[str] = []

    for _, player in bench.iterrows():
        rows.append(
            f"- {int(player['bench_order'])}. "
            f"{player['player_name']} "
            f"({player['team_name']}, "
            f"{player['position']}) | "
            f"{_format_points(player['next_gameweek_projected_points'])} pts"
        )

    return rows


def _lineup_rows_html(
    starting_xi: pd.DataFrame,
) -> str:
    """Build HTML rows for the starting XI."""

    rows: list[str] = []

    for line in _lineup_rows_text(
        starting_xi
    ):
        rows.append(
            f"<li>{escape(line[2:])}</li>"
        )

    return "\n".join(
        rows
    )


def _bench_rows_html(
    bench: pd.DataFrame,
) -> str:
    """Build HTML rows for substitutes."""

    rows: list[str] = []

    for line in _bench_rows_text(
        bench
    ):
        rows.append(
            f"<li>{escape(line[2:])}</li>"
        )

    return "\n".join(
        rows
    )


def _transfer_actions_html(
    transfer_plan: TransferPlan,
) -> str:
    """Build HTML transfer instructions."""

    return "\n".join(
        f"<li>{escape(action)}</li>"
        for action in _transfer_actions(
            transfer_plan
        )
    )


def build_weekly_report(
    deadline_intelligence: DeadlineIntelligence,
    squad_value: SquadValueResult,
    transfer_plan: TransferPlan,
    starting_xi: pd.DataFrame,
    bench: pd.DataFrame,
    chip_recommendation: ChipRecommendation,
    lifecycle_status: str,
) -> WeeklyReport:
    """Build plain-text and HTML weekly decision reports."""

    lineup = _validate_starting_xi(
        starting_xi
    )

    substitutes = _validate_bench(
        bench
    )

    gameweek = _extract_gameweek(
        lineup
    )

    captain = _captain_row(
        lineup
    )

    vice_captain = _vice_captain_row(
        lineup
    )

    formation = _formation(
        lineup
    )

    starting_projection = float(
        lineup[
            "next_gameweek_projected_points"
        ].sum()
    )

    captain_bonus = float(
        captain[
            "next_gameweek_projected_points"
        ]
    )

    projected_team_total = (
        starting_projection
        + captain_bonus
    )

    bench_projection = float(
        substitutes[
            "next_gameweek_projected_points"
        ].sum()
    )

    transfer_actions = _transfer_actions(
        transfer_plan
    )

    text_lines = [
        "=" * 68,
        f"PROJECT AIRAOLA | GAMEWEEK {gameweek} DECISION REPORT",
        "Data. Decisions. Domination.",
        "=" * 68,
        "",
        "DEADLINE",
        f"Official deadline: {_deadline_text(deadline_intelligence)}",
        (
            "Hours remaining: "
            + (
                f"{deadline_intelligence.hours_until_deadline:.2f}"
                if deadline_intelligence.hours_until_deadline
                is not None
                else "unavailable"
            )
        ),
        (
            "Analysis window: "
            f"{deadline_intelligence.deadline_window}"
        ),
        (
            "Season-clock status: "
            f"{deadline_intelligence.state_status}"
        ),
        "",
        "FINANCE",
        (
            "Squad purchase value: "
            f"{_format_money(squad_value.purchase_value)}"
        ),
        (
            "Current market value: "
            f"{_format_money(squad_value.current_market_value)}"
        ),
        (
            "Official selling value: "
            f"{_format_money(squad_value.selling_value)}"
        ),
        (
            "Money in bank: "
            f"{_format_money(squad_value.bank)}"
        ),
        (
            "Available budget: "
            f"{_format_money(squad_value.available_budget)}"
        ),
        "",
        "TRANSFER DECISION",
        f"Decision: {transfer_plan.decision}",
        (
            "Recommendation strength: "
            f"{transfer_plan.recommendation_strength}"
        ),
        (
            "Gross projected gain: "
            f"{_format_points(transfer_plan.gross_projected_gain, True)}"
        ),
        (
            "Net strategic gain: "
            f"{_format_points(transfer_plan.net_strategic_gain, True)}"
        ),
        (
            "Hit cost: "
            f"{_format_points(transfer_plan.hit_cost)}"
        ),
        f"Reason: {transfer_plan.reason}",
        "",
        "MANUAL TRANSFER ACTIONS",
        *[
            f"- {action}"
            for action in transfer_actions
        ],
        "",
        "STARTING XI",
        f"Formation: {formation}",
        *(
            _lineup_rows_text(
                lineup
            )
        ),
        "",
        (
            "Captain: "
            f"{captain['player_name']} "
            f"({_format_points(captain['next_gameweek_projected_points'])} pts)"
        ),
        (
            "Vice-captain: "
            f"{vice_captain['player_name']} "
            f"({_format_points(vice_captain['next_gameweek_projected_points'])} pts)"
        ),
        (
            "Projected XI points before captaincy: "
            f"{_format_points(starting_projection)}"
        ),
        (
            "Projected team total with captaincy: "
            f"{_format_points(projected_team_total)}"
        ),
        "",
        "BENCH",
        *(
            _bench_rows_text(
                substitutes
            )
        ),
        (
            "Projected bench points: "
            f"{_format_points(bench_projection)}"
        ),
        "",
        "CHIP DECISION",
        f"Decision: {chip_recommendation.decision}",
        (
            "Recommendation strength: "
            f"{chip_recommendation.recommendation_strength}"
        ),
        (
            "Projected gain: "
            f"{_format_points(chip_recommendation.projected_gain, True)}"
        ),
        (
            "Adjusted gain: "
            f"{_format_points(chip_recommendation.adjusted_gain, True)}"
        ),
        f"Reason: {chip_recommendation.reason}",
        "",
        "MANUAL CHIP ACTION",
        f"- {_chip_action(chip_recommendation)}",
        "",
        "LIFECYCLE",
        (
            "Lifecycle status: "
            f"{_normalise_text(lifecycle_status)}"
        ),
        (
            "Deadline recommendation: "
            f"{deadline_intelligence.recommendation}"
        ),
        "",
        "FINAL CHECKLIST",
        "- Complete the listed transfers in FPL.",
        (
            "- Set captain to "
            f"{captain['player_name']}."
        ),
        (
            "- Set vice-captain to "
            f"{vice_captain['player_name']}."
        ),
        "- Arrange the bench in the listed order.",
        f"- {_chip_action(chip_recommendation)}",
        "- Confirm all changes before the official deadline.",
        "",
    ]

    text_content = "\n".join(
        text_lines
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Project Airaola Gameweek {gameweek} Report</title>
</head>
<body>
    <main>
        <h1>Project Airaola</h1>
        <p><strong>Data. Decisions. Domination.</strong></p>
        <h2>Gameweek {gameweek} Decision Report</h2>

        <h3>Deadline</h3>
        <p>
            <strong>Official deadline:</strong>
            {escape(_deadline_text(deadline_intelligence))}
        </p>
        <p>
            <strong>Hours remaining:</strong>
            {
                f"{deadline_intelligence.hours_until_deadline:.2f}"
                if deadline_intelligence.hours_until_deadline is not None
                else "unavailable"
            }
        </p>
        <p>
            <strong>Analysis window:</strong>
            {escape(deadline_intelligence.deadline_window)}
        </p>
        <p>
            <strong>Season-clock status:</strong>
            {escape(deadline_intelligence.state_status)}
        </p>

        <h3>Finance</h3>
        <ul>
            <li>Purchase value: {escape(_format_money(squad_value.purchase_value))}</li>
            <li>Market value: {escape(_format_money(squad_value.current_market_value))}</li>
            <li>Selling value: {escape(_format_money(squad_value.selling_value))}</li>
            <li>Money in bank: {escape(_format_money(squad_value.bank))}</li>
            <li>Available budget: {escape(_format_money(squad_value.available_budget))}</li>
        </ul>

        <h3>Transfer Decision</h3>
        <p><strong>Decision:</strong> {escape(str(transfer_plan.decision))}</p>
        <p>
            <strong>Recommendation strength:</strong>
            {escape(str(transfer_plan.recommendation_strength))}
        </p>
        <p>
            <strong>Net strategic gain:</strong>
            {escape(_format_points(transfer_plan.net_strategic_gain, True))}
        </p>
        <p>{escape(str(transfer_plan.reason))}</p>

        <h4>Manual Transfer Actions</h4>
        <ul>
            {_transfer_actions_html(transfer_plan)}
        </ul>

        <h3>Starting XI</h3>
        <p><strong>Formation:</strong> {escape(formation)}</p>
        <ul>
            {_lineup_rows_html(lineup)}
        </ul>
        <p>
            <strong>Captain:</strong>
            {escape(str(captain["player_name"]))}
        </p>
        <p>
            <strong>Vice-captain:</strong>
            {escape(str(vice_captain["player_name"]))}
        </p>
        <p>
            <strong>Projected total with captaincy:</strong>
            {escape(_format_points(projected_team_total))}
        </p>

        <h3>Bench</h3>
        <ol>
            {_bench_rows_html(substitutes)}
        </ol>
        <p>
            <strong>Projected bench points:</strong>
            {escape(_format_points(bench_projection))}
        </p>

        <h3>Chip Decision</h3>
        <p>
            <strong>Decision:</strong>
            {escape(str(chip_recommendation.decision))}
        </p>
        <p>
            <strong>Recommendation strength:</strong>
            {escape(str(chip_recommendation.recommendation_strength))}
        </p>
        <p>
            <strong>Adjusted gain:</strong>
            {escape(_format_points(chip_recommendation.adjusted_gain, True))}
        </p>
        <p>{escape(str(chip_recommendation.reason))}</p>
        <p>
            <strong>Manual action:</strong>
            {escape(_chip_action(chip_recommendation))}
        </p>

        <h3>Lifecycle</h3>
        <p>
            <strong>Status:</strong>
            {escape(_normalise_text(lifecycle_status))}
        </p>
        <p>{escape(deadline_intelligence.recommendation)}</p>

        <h3>Final Checklist</h3>
        <ul>
            <li>Complete the listed transfers in FPL.</li>
            <li>Set captain to {escape(str(captain["player_name"]))}.</li>
            <li>Set vice-captain to {escape(str(vice_captain["player_name"]))}.</li>
            <li>Arrange the bench in the listed order.</li>
            <li>{escape(_chip_action(chip_recommendation))}</li>
            <li>Confirm all changes before the official deadline.</li>
        </ul>
    </main>
</body>
</html>
"""

    return WeeklyReport(
        gameweek=gameweek,
        text_content=text_content,
        html_content=html_content,
    )


def save_weekly_report(
    report: WeeklyReport,
    reports_path: Path = REPORTS_PATH,
) -> WeeklyReport:
    """Save text and HTML versions of a weekly report."""

    reports_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    text_path = (
        reports_path
        / f"gameweek_{report.gameweek}_report.txt"
    )

    html_path = (
        reports_path
        / f"gameweek_{report.gameweek}_report.html"
    )

    text_path.write_text(
        report.text_content,
        encoding="utf-8",
    )

    html_path.write_text(
        report.html_content,
        encoding="utf-8",
    )

    return WeeklyReport(
        gameweek=report.gameweek,
        text_content=report.text_content,
        html_content=report.html_content,
        text_path=text_path,
        html_path=html_path,
    )