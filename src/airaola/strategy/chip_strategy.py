from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


NO_CHIP = "NO CHIP"
TRIPLE_CAPTAIN = "TRIPLE CAPTAIN"
BENCH_BOOST = "BENCH BOOST"

FIRST_HALF_FINAL_GAMEWEEK = 19
SECOND_HALF_START_GAMEWEEK = 20

MINIMUM_TRIPLE_CAPTAIN_GAIN = 8.0
MINIMUM_BENCH_BOOST_GAIN = 14.0

EXPIRY_PRESSURE_START_GAMEWEEK = 16
EXPIRY_PRESSURE_PER_GAMEWEEK = 1.5

MINIMUM_CAPTAIN_MINUTES_SECURITY = 0.85
MINIMUM_BENCH_PLAYER_SECURITY = 0.70


@dataclass(frozen=True)
class ChipCandidate:
    """Represent one evaluated chip option."""

    chip_name: str
    available: bool
    eligible: bool

    projected_gain: float
    adjusted_gain: float
    threshold: float

    recommendation_strength: str
    reason: str


@dataclass(frozen=True)
class ChipRecommendation:
    """Represent Airaola's selected chip strategy."""

    decision: str
    chip_period: str

    current_gameweek: int
    projected_gain: float
    adjusted_gain: float
    execution_threshold: float

    recommendation_strength: str
    reason: str

    captain_name: str | None
    captain_projected_points: float

    bench_projected_points: float
    bench_players: tuple[str, ...]

    candidates: tuple[ChipCandidate, ...]


def _safe_numeric(
    dataframe: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """Return one safely converted numeric column."""

    if column not in dataframe.columns:
        return pd.Series(
            default,
            index=dataframe.index,
            dtype=float,
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).fillna(default)


def _normalise_chip_name(
    chip_name: str,
) -> str:
    """Convert state names into a consistent display form."""

    return (
        chip_name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _current_chip_period(
    current_gameweek: int,
) -> str:
    """Return the active half-season chip period."""

    if current_gameweek <= FIRST_HALF_FINAL_GAMEWEEK:
        return "first_half"

    return "second_half"


def _chip_is_available(
    chips: dict[str, Any],
    chip_period: str,
    chip_name: str,
) -> bool:
    """
    Read chip availability from either the new nested state format
    or the previous flat state format.
    """

    normalised_name = _normalise_chip_name(
        chip_name
    )

    period_data = chips.get(
        chip_period,
    )

    if isinstance(period_data, dict):
        return bool(
            period_data.get(
                normalised_name,
                False,
            )
        )

    legacy_keys = {
        "triple_captain": "triple_captain",
        "bench_boost": "bench_boost",
        "free_hit": "free_hit",
        "wildcard": (
            "wildcard_1"
            if chip_period == "first_half"
            else "wildcard_2"
        ),
    }

    legacy_key = legacy_keys.get(
        normalised_name
    )

    if legacy_key is None:
        return False

    return bool(
        chips.get(
            legacy_key,
            False,
        )
    )


def _expiry_pressure(
    current_gameweek: int,
) -> float:
    """
    Increase the value of using an available first-half chip as
    the Gameweek 19 expiry approaches.
    """

    if not (
        EXPIRY_PRESSURE_START_GAMEWEEK
        <= current_gameweek
        <= FIRST_HALF_FINAL_GAMEWEEK
    ):
        return 0.0

    weeks_into_pressure_window = (
        current_gameweek
        - EXPIRY_PRESSURE_START_GAMEWEEK
        + 1
    )

    return (
        weeks_into_pressure_window
        * EXPIRY_PRESSURE_PER_GAMEWEEK
    )


def _recommendation_strength(
    adjusted_gain: float,
    threshold: float,
) -> str:
    """Classify a chip candidate's strategic value."""

    margin = adjusted_gain - threshold

    if margin >= 8.0:
        return "EXCEPTIONAL"

    if margin >= 4.0:
        return "VERY STRONG"

    if margin >= 2.0:
        return "STRONG"

    if margin >= 0.0:
        return "MODERATE"

    return "HOLD"


def _validate_inputs(
    starting_xi: pd.DataFrame,
    bench: pd.DataFrame,
    current_gameweek: int,
    chips: dict[str, Any],
) -> None:
    """Validate the chip engine's required inputs."""

    if len(starting_xi) != 11:
        raise ValueError(
            "Chip strategy requires exactly "
            "11 starting players."
        )

    if len(bench) != 4:
        raise ValueError(
            "Chip strategy requires exactly "
            "four substitutes."
        )

    if current_gameweek < 1:
        raise ValueError(
            "Current Gameweek must be at least 1."
        )

    if not isinstance(chips, dict):
        raise ValueError(
            "Chip availability must be supplied "
            "as a dictionary."
        )

    required_xi_columns = {
        "player_name",
        "next_gameweek_projected_points",
        "minutes_security",
        "is_captain",
    }

    missing_xi_columns = (
        required_xi_columns.difference(
            starting_xi.columns
        )
    )

    if missing_xi_columns:
        raise ValueError(
            "Starting XI is missing chip columns: "
            + ", ".join(
                sorted(missing_xi_columns)
            )
        )

    required_bench_columns = {
        "player_name",
        "position",
        "next_gameweek_projected_points",
        "minutes_security",
    }

    missing_bench_columns = (
        required_bench_columns.difference(
            bench.columns
        )
    )

    if missing_bench_columns:
        raise ValueError(
            "Bench is missing chip columns: "
            + ", ".join(
                sorted(missing_bench_columns)
            )
        )


def _evaluate_triple_captain(
    starting_xi: pd.DataFrame,
    current_gameweek: int,
    chips: dict[str, Any],
    chip_period: str,
) -> ChipCandidate:
    """Evaluate the projected value of Triple Captain."""

    available = _chip_is_available(
        chips=chips,
        chip_period=chip_period,
        chip_name="triple_captain",
    )

    captain_rows = starting_xi[
        starting_xi["is_captain"].astype(bool)
    ].copy()

    if captain_rows.empty:
        return ChipCandidate(
            chip_name=TRIPLE_CAPTAIN,
            available=available,
            eligible=False,
            projected_gain=0.0,
            adjusted_gain=0.0,
            threshold=MINIMUM_TRIPLE_CAPTAIN_GAIN,
            recommendation_strength="HOLD",
            reason=(
                "No captain was identified in the "
                "selected starting XI."
            ),
        )

    captain = captain_rows.iloc[0]

    projected_points = float(
        captain[
            "next_gameweek_projected_points"
        ]
    )

    minutes_security = float(
        captain["minutes_security"]
    )

    eligible = (
        available
        and minutes_security
        >= MINIMUM_CAPTAIN_MINUTES_SECURITY
    )

    pressure = (
        _expiry_pressure(current_gameweek)
        if chip_period == "first_half"
        else 0.0
    )

    adjusted_gain = (
        projected_points
        + pressure
    )

    if not available:
        reason = (
            "Triple Captain is not available in "
            "the active chip period."
        )
    elif (
        minutes_security
        < MINIMUM_CAPTAIN_MINUTES_SECURITY
    ):
        reason = (
            "The selected captain's minutes security "
            f"is only {minutes_security:.3f}, below "
            "Airaola's chip-safety requirement."
        )
    else:
        reason = (
            "Triple Captain would add one additional "
            "copy of the captain's projected "
            f"{projected_points:.2f} points."
        )

        if pressure > 0:
            reason += (
                " First-half expiry pressure adds "
                f"{pressure:.2f} strategic points."
            )

    return ChipCandidate(
        chip_name=TRIPLE_CAPTAIN,
        available=available,
        eligible=eligible,
        projected_gain=round(
            projected_points,
            2,
        ),
        adjusted_gain=round(
            adjusted_gain,
            2,
        ),
        threshold=MINIMUM_TRIPLE_CAPTAIN_GAIN,
        recommendation_strength=(
            _recommendation_strength(
                adjusted_gain=adjusted_gain,
                threshold=(
                    MINIMUM_TRIPLE_CAPTAIN_GAIN
                ),
            )
            if eligible
            else "HOLD"
        ),
        reason=reason,
    )


def _evaluate_bench_boost(
    bench: pd.DataFrame,
    current_gameweek: int,
    chips: dict[str, Any],
    chip_period: str,
) -> ChipCandidate:
    """Evaluate the projected value of Bench Boost."""

    available = _chip_is_available(
        chips=chips,
        chip_period=chip_period,
        chip_name="bench_boost",
    )

    evaluated_bench = bench.copy()

    evaluated_bench[
        "next_gameweek_projected_points"
    ] = _safe_numeric(
        evaluated_bench,
        "next_gameweek_projected_points",
    )

    evaluated_bench["minutes_security"] = (
        _safe_numeric(
            evaluated_bench,
            "minutes_security",
        )
    )

    projected_gain = float(
        evaluated_bench[
            "next_gameweek_projected_points"
        ].sum()
    )

    secure_players = int(
        (
            evaluated_bench["minutes_security"]
            >= MINIMUM_BENCH_PLAYER_SECURITY
        ).sum()
    )

    goalkeeper_rows = evaluated_bench[
        evaluated_bench["position"] == "GKP"
    ]

    goalkeeper_secure = (
        len(goalkeeper_rows) == 1
        and float(
            goalkeeper_rows.iloc[0][
                "minutes_security"
            ]
        )
        >= MINIMUM_BENCH_PLAYER_SECURITY
    )

    eligible = (
        available
        and secure_players == 4
        and goalkeeper_secure
    )

    pressure = (
        _expiry_pressure(current_gameweek)
        if chip_period == "first_half"
        else 0.0
    )

    adjusted_gain = (
        projected_gain
        + pressure
    )

    if not available:
        reason = (
            "Bench Boost is not available in "
            "the active chip period."
        )
    elif secure_players < 4:
        reason = (
            f"Only {secure_players} of four bench "
            "players clear Airaola's minutes-security "
            "requirement."
        )
    elif not goalkeeper_secure:
        reason = (
            "The substitute goalkeeper does not clear "
            "Airaola's minutes-security requirement."
        )
    else:
        reason = (
            "Bench Boost would add the projected "
            f"{projected_gain:.2f} points from all "
            "four substitutes."
        )

        if pressure > 0:
            reason += (
                " First-half expiry pressure adds "
                f"{pressure:.2f} strategic points."
            )

    return ChipCandidate(
        chip_name=BENCH_BOOST,
        available=available,
        eligible=eligible,
        projected_gain=round(
            projected_gain,
            2,
        ),
        adjusted_gain=round(
            adjusted_gain,
            2,
        ),
        threshold=MINIMUM_BENCH_BOOST_GAIN,
        recommendation_strength=(
            _recommendation_strength(
                adjusted_gain=adjusted_gain,
                threshold=(
                    MINIMUM_BENCH_BOOST_GAIN
                ),
            )
            if eligible
            else "HOLD"
        ),
        reason=reason,
    )


def recommend_chip_strategy(
    starting_xi: pd.DataFrame,
    bench: pd.DataFrame,
    current_gameweek: int,
    chips: dict[str, Any],
) -> ChipRecommendation:
    """
    Evaluate available one-Gameweek scoring chips.

    This first v0.1.15 implementation evaluates No Chip,
    Triple Captain and Bench Boost. Free Hit and Wildcard are
    intentionally deferred until temporary and permanent squad
    re-optimisation are integrated.
    """

    _validate_inputs(
        starting_xi=starting_xi,
        bench=bench,
        current_gameweek=current_gameweek,
        chips=chips,
    )

    chip_period = _current_chip_period(
        current_gameweek
    )

    triple_captain = _evaluate_triple_captain(
        starting_xi=starting_xi,
        current_gameweek=current_gameweek,
        chips=chips,
        chip_period=chip_period,
    )

    bench_boost = _evaluate_bench_boost(
        bench=bench,
        current_gameweek=current_gameweek,
        chips=chips,
        chip_period=chip_period,
    )

    candidates = (
        triple_captain,
        bench_boost,
    )

    executable_candidates = [
        candidate
        for candidate in candidates
        if (
            candidate.available
            and candidate.eligible
            and candidate.adjusted_gain
            >= candidate.threshold
        )
    ]

    captain_rows = starting_xi[
        starting_xi["is_captain"].astype(bool)
    ]

    if captain_rows.empty:
        captain_name = None
        captain_projected_points = 0.0
    else:
        captain = captain_rows.iloc[0]

        captain_name = str(
            captain["player_name"]
        )

        captain_projected_points = float(
            captain[
                "next_gameweek_projected_points"
            ]
        )

    bench_players = tuple(
        str(player_name)
        for player_name in bench[
            "player_name"
        ].tolist()
    )

    bench_projected_points = float(
        _safe_numeric(
            bench,
            "next_gameweek_projected_points",
        ).sum()
    )

    if not executable_candidates:
        best_candidate = max(
            candidates,
            key=lambda candidate: (
                candidate.adjusted_gain
                - candidate.threshold
            ),
        )

        return ChipRecommendation(
            decision=NO_CHIP,
            chip_period=chip_period,
            current_gameweek=current_gameweek,
            projected_gain=0.0,
            adjusted_gain=0.0,
            execution_threshold=0.0,
            recommendation_strength="HOLD",
            reason=(
                "No available chip clears Airaola's "
                "execution threshold. The strongest "
                "rejected option was "
                f"{best_candidate.chip_name} at "
                f"{best_candidate.adjusted_gain:.2f} "
                "adjusted points against a "
                f"{best_candidate.threshold:.2f} "
                "threshold."
            ),
            captain_name=captain_name,
            captain_projected_points=round(
                captain_projected_points,
                2,
            ),
            bench_projected_points=round(
                bench_projected_points,
                2,
            ),
            bench_players=bench_players,
            candidates=candidates,
        )

    selected = max(
        executable_candidates,
        key=lambda candidate: (
            candidate.adjusted_gain
            - candidate.threshold,
            candidate.adjusted_gain,
        ),
    )

    return ChipRecommendation(
        decision=selected.chip_name,
        chip_period=chip_period,
        current_gameweek=current_gameweek,
        projected_gain=selected.projected_gain,
        adjusted_gain=selected.adjusted_gain,
        execution_threshold=selected.threshold,
        recommendation_strength=(
            selected.recommendation_strength
        ),
        reason=selected.reason,
        captain_name=captain_name,
        captain_projected_points=round(
            captain_projected_points,
            2,
        ),
        bench_projected_points=round(
            bench_projected_points,
            2,
        ),
        bench_players=bench_players,
        candidates=candidates,
    )