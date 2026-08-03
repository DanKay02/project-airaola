from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from airaola.finance.price_engine import (
    SquadValueResult,
)
from airaola.optimisation.transfer_planner import (
    TransferMove,
    TransferPlan,
)
from airaola.reporting.weekly_report import (
    build_weekly_report,
    save_weekly_report,
)
from airaola.strategy.chip_strategy import (
    ChipRecommendation,
)


def _deadline_intelligence() -> SimpleNamespace:
    """Build synthetic deadline intelligence."""

    deadline = datetime(
        2026,
        8,
        21,
        18,
        30,
        tzinfo=timezone.utc,
    )

    return SimpleNamespace(
        next_deadline_gameweek=1,
        next_deadline_local=deadline,
        hours_until_deadline=24.5,
        deadline_window="ON TIME",
        state_status="ALIGNED",
        recommendation=(
            "State is aligned with the official FPL season clock."
        ),
    )


def _squad_value() -> SquadValueResult:
    """Build synthetic squad finances."""

    return SquadValueResult(
        purchase_value=99.5,
        current_market_value=100.2,
        selling_value=99.9,
        bank=0.5,
        available_budget=100.4,
        market_change=0.7,
        realised_change=0.4,
        unrealised_profit=0.8,
        retained_profit=0.5,
        lost_value=0.1,
        player_prices=(),
    )


def _starting_xi() -> pd.DataFrame:
    """Build a legal synthetic 3-5-2 starting XI."""

    players = [
        {
            "id": 1,
            "player_name": "Goalkeeper One",
            "team_name": "Alpha",
            "position": "GKP",
            "next_gameweek": 1,
            "next_gameweek_projected_points": 4.0,
            "is_captain": False,
            "is_vice_captain": False,
        },
        {
            "id": 2,
            "player_name": "Defender Captain",
            "team_name": "Bravo",
            "position": "DEF",
            "next_gameweek": 1,
            "next_gameweek_projected_points": 7.0,
            "is_captain": True,
            "is_vice_captain": False,
        },
        {
            "id": 3,
            "player_name": "Defender Two",
            "team_name": "Charlie",
            "position": "DEF",
            "next_gameweek": 1,
            "next_gameweek_projected_points": 5.0,
            "is_captain": False,
            "is_vice_captain": False,
        },
        {
            "id": 4,
            "player_name": "Defender Three",
            "team_name": "Delta",
            "position": "DEF",
            "next_gameweek": 1,
            "next_gameweek_projected_points": 4.5,
            "is_captain": False,
            "is_vice_captain": False,
        },
        {
            "id": 5,
            "player_name": "Midfielder Vice",
            "team_name": "Echo",
            "position": "MID",
            "next_gameweek": 1,
            "next_gameweek_projected_points": 6.5,
            "is_captain": False,
            "is_vice_captain": True,
        },
        {
            "id": 6,
            "player_name": "Midfielder Two",
            "team_name": "Foxtrot",
            "position": "MID",
            "next_gameweek": 1,
            "next_gameweek_projected_points": 5.5,
            "is_captain": False,
            "is_vice_captain": False,
        },
        {
            "id": 7,
            "player_name": "Midfielder Three",
            "team_name": "Golf",
            "position": "MID",
            "next_gameweek": 1,
            "next_gameweek_projected_points": 5.0,
            "is_captain": False,
            "is_vice_captain": False,
        },
        {
            "id": 8,
            "player_name": "Midfielder Four",
            "team_name": "Hotel",
            "position": "MID",
            "next_gameweek": 1,
            "next_gameweek_projected_points": 4.0,
            "is_captain": False,
            "is_vice_captain": False,
        },
        {
            "id": 9,
            "player_name": "Midfielder Five",
            "team_name": "India",
            "position": "MID",
            "next_gameweek": 1,
            "next_gameweek_projected_points": 3.5,
            "is_captain": False,
            "is_vice_captain": False,
        },
        {
            "id": 10,
            "player_name": "Forward One",
            "team_name": "Juliet",
            "position": "FWD",
            "next_gameweek": 1,
            "next_gameweek_projected_points": 6.0,
            "is_captain": False,
            "is_vice_captain": False,
        },
        {
            "id": 11,
            "player_name": "Forward Two",
            "team_name": "Kilo",
            "position": "FWD",
            "next_gameweek": 1,
            "next_gameweek_projected_points": 5.0,
            "is_captain": False,
            "is_vice_captain": False,
        },
    ]

    return pd.DataFrame(
        players
    )


def _bench() -> pd.DataFrame:
    """Build a synthetic ordered bench."""

    return pd.DataFrame(
        [
            {
                "id": 12,
                "player_name": "Bench Forward",
                "team_name": "Lima",
                "position": "FWD",
                "bench_order": 1,
                "next_gameweek_projected_points": 3.0,
            },
            {
                "id": 13,
                "player_name": "Bench Defender One",
                "team_name": "Mike",
                "position": "DEF",
                "bench_order": 2,
                "next_gameweek_projected_points": 2.5,
            },
            {
                "id": 14,
                "player_name": "Bench Defender Two",
                "team_name": "November",
                "position": "DEF",
                "bench_order": 3,
                "next_gameweek_projected_points": 2.0,
            },
            {
                "id": 15,
                "player_name": "Reserve Goalkeeper",
                "team_name": "Oscar",
                "position": "GKP",
                "bench_order": 4,
                "next_gameweek_projected_points": 1.5,
            },
        ]
    )


def _roll_transfer_plan() -> TransferPlan:
    """Build a synthetic ROLL recommendation."""

    return TransferPlan(
        decision="ROLL",
        free_transfers_before=1,
        free_transfers_next_gameweek=2,
        bank_before=0.5,
        bank_after=0.5,
        gross_projected_gain=0.8,
        net_strategic_gain=0.05,
        execution_threshold=1.5,
        recommendation_strength="HOLD",
        reason=(
            "The strongest transfer does not clear "
            "the execution threshold."
        ),
    )


def _execute_transfer_plan() -> TransferPlan:
    """Build a synthetic one-transfer recommendation."""

    move = TransferMove(
        player_out_id=20,
        player_out_name="Outgoing Player",
        player_in_id=21,
        player_in_name="Incoming Player",
        position="MID",
        selling_price=6.5,
        purchase_price=6.7,
        projected_gain=3.0,
        next_gameweek_gain=1.2,
    )

    return TransferPlan(
        decision="EXECUTE",
        transfers=(
            move,
        ),
        free_transfers_before=1,
        transfers_used=1,
        free_transfers_spent=1,
        hit_transfers=0,
        hit_cost=0.0,
        gross_projected_gain=3.0,
        next_gameweek_gain=1.2,
        transfer_bank_cost=0.75,
        net_strategic_gain=2.25,
        execution_threshold=1.5,
        bank_before=0.5,
        bank_after=0.3,
        free_transfers_next_gameweek=1,
        recommendation_strength="MODERATE",
        reason=(
            "The proposed transfer clears the "
            "execution threshold."
        ),
    )


def _chip_recommendation(
    decision: str = "NO CHIP",
) -> ChipRecommendation:
    """Build a synthetic chip recommendation."""

    return ChipRecommendation(
        decision=decision,
        chip_period="first_half",
        current_gameweek=1,
        projected_gain=0.0,
        adjusted_gain=0.0,
        execution_threshold=0.0,
        recommendation_strength="HOLD",
        reason=(
            "No chip clears the execution threshold."
        ),
        captain_name="Defender Captain",
        captain_projected_points=7.0,
        bench_projected_points=9.0,
        bench_players=(
            "Bench Forward",
            "Bench Defender One",
            "Bench Defender Two",
            "Reserve Goalkeeper",
        ),
        candidates=(),
    )


def _build_report(
    transfer_plan: TransferPlan | None = None,
    chip_decision: str = "NO CHIP",
):
    """Build one report using shared synthetic data."""

    return build_weekly_report(
        deadline_intelligence=(
            _deadline_intelligence()
        ),
        squad_value=_squad_value(),
        transfer_plan=(
            transfer_plan
            if transfer_plan is not None
            else _roll_transfer_plan()
        ),
        starting_xi=_starting_xi(),
        bench=_bench(),
        chip_recommendation=(
            _chip_recommendation(
                chip_decision
            )
        ),
        lifecycle_status="OPEN",
    )


def test_builds_text_report() -> None:
    report = _build_report()

    assert report.gameweek == 1
    assert (
        "PROJECT AIRAOLA | GAMEWEEK 1 DECISION REPORT"
        in report.text_content
    )
    assert "Data. Decisions. Domination." in report.text_content
    assert "TRANSFER DECISION" in report.text_content
    assert "STARTING XI" in report.text_content
    assert "BENCH" in report.text_content
    assert "CHIP DECISION" in report.text_content


def test_builds_html_report() -> None:
    report = _build_report()

    assert "<!DOCTYPE html>" in report.html_content
    assert "<h1>Project Airaola</h1>" in report.html_content
    assert "Gameweek 1 Decision Report" in report.html_content
    assert "<h3>Transfer Decision</h3>" in report.html_content
    assert "<h3>Starting XI</h3>" in report.html_content


def test_captain_and_vice_captain_are_reported() -> None:
    report = _build_report()

    assert (
        "Captain: Defender Captain"
        in report.text_content
    )

    assert (
        "Vice-captain: Midfielder Vice"
        in report.text_content
    )

    assert (
        "Defender Captain (Bravo) [C]"
        in report.text_content
    )

    assert (
        "Midfielder Vice (Echo) [VC]"
        in report.text_content
    )


def test_formation_is_reported() -> None:
    report = _build_report()

    assert "Formation: 3-5-2" in report.text_content


def test_bench_is_reported_in_order() -> None:
    report = _build_report()

    first = report.text_content.index(
        "1. Bench Forward"
    )
    second = report.text_content.index(
        "2. Bench Defender One"
    )
    third = report.text_content.index(
        "3. Bench Defender Two"
    )
    fourth = report.text_content.index(
        "4. Reserve Goalkeeper"
    )

    assert first < second < third < fourth


def test_roll_transfer_instruction_is_reported() -> None:
    report = _build_report()

    assert "Decision: ROLL" in report.text_content
    assert "Roll the free transfer." in report.text_content
    assert (
        "Expected free transfers next Gameweek: 2"
        in report.text_content
    )


def test_execute_transfer_instruction_is_reported() -> None:
    report = _build_report(
        transfer_plan=_execute_transfer_plan()
    )

    assert "Decision: EXECUTE" in report.text_content
    assert "Sell Outgoing Player" in report.text_content
    assert "buy Incoming Player" in report.text_content
    assert "£6.5m" in report.text_content
    assert "£6.7m" in report.text_content


def test_no_chip_instruction_is_reported() -> None:
    report = _build_report()

    assert "Decision: NO CHIP" in report.text_content
    assert "Do not activate a chip." in report.text_content


def test_active_chip_instruction_is_reported() -> None:
    report = _build_report(
        chip_decision="TRIPLE CAPTAIN"
    )

    assert (
        "Activate Triple Captain."
        in report.text_content
    )


def test_report_files_are_saved(
    tmp_path: Path,
) -> None:
    report = _build_report()

    saved_report = save_weekly_report(
        report=report,
        reports_path=tmp_path,
    )

    assert saved_report.text_path is not None
    assert saved_report.html_path is not None

    assert saved_report.text_path.exists()
    assert saved_report.html_path.exists()

    assert (
        saved_report.text_path.name
        == "gameweek_1_report.txt"
    )

    assert (
        saved_report.html_path.name
        == "gameweek_1_report.html"
    )

    assert (
        saved_report.text_path.read_text(
            encoding="utf-8"
        )
        == report.text_content
    )

    assert (
        saved_report.html_path.read_text(
            encoding="utf-8"
        )
        == report.html_content
    )


def test_report_build_does_not_write_files(
    tmp_path: Path,
) -> None:
    _build_report()

    assert list(
        tmp_path.iterdir()
    ) == []


def test_invalid_starting_xi_size_is_rejected() -> None:
    starting_xi = (
        _starting_xi()
        .iloc[:10]
        .copy()
    )

    with pytest.raises(
        ValueError,
        match="exactly 11 starters",
    ):
        build_weekly_report(
            deadline_intelligence=(
                _deadline_intelligence()
            ),
            squad_value=_squad_value(),
            transfer_plan=_roll_transfer_plan(),
            starting_xi=starting_xi,
            bench=_bench(),
            chip_recommendation=(
                _chip_recommendation()
            ),
            lifecycle_status="OPEN",
        )


def test_multiple_captains_are_rejected() -> None:
    starting_xi = _starting_xi()

    starting_xi.loc[
        starting_xi["id"] == 3,
        "is_captain",
    ] = True

    with pytest.raises(
        ValueError,
        match="exactly one captain",
    ):
        build_weekly_report(
            deadline_intelligence=(
                _deadline_intelligence()
            ),
            squad_value=_squad_value(),
            transfer_plan=_roll_transfer_plan(),
            starting_xi=starting_xi,
            bench=_bench(),
            chip_recommendation=(
                _chip_recommendation()
            ),
            lifecycle_status="OPEN",
        )


def test_invalid_bench_order_is_rejected() -> None:
    bench = _bench()

    bench.loc[
        bench["bench_order"] == 4,
        "bench_order",
    ] = 3

    with pytest.raises(
        ValueError,
        match="positions 1 to 4",
    ):
        build_weekly_report(
            deadline_intelligence=(
                _deadline_intelligence()
            ),
            squad_value=_squad_value(),
            transfer_plan=_roll_transfer_plan(),
            starting_xi=_starting_xi(),
            bench=bench,
            chip_recommendation=(
                _chip_recommendation()
            ),
            lifecycle_status="OPEN",
        )