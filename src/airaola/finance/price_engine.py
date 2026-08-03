from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import Any

import pandas as pd


FPL_PRICE_INCREMENT = Decimal("0.1")
FPL_INITIAL_BUDGET = Decimal("100.0")


@dataclass(frozen=True)
class PlayerPriceResult:
    """Describe the finances of one owned FPL player."""

    player_id: int
    player_name: str

    purchase_price: float
    current_price: float
    selling_price: float

    market_change: float
    realised_change: float
    unrealised_profit: float
    retained_profit: float
    lost_value: float


@dataclass(frozen=True)
class SquadValueResult:
    """Describe the financial value of a complete squad."""

    purchase_value: float
    current_market_value: float
    selling_value: float
    bank: float
    available_budget: float

    market_change: float
    realised_change: float
    unrealised_profit: float
    retained_profit: float
    lost_value: float

    player_prices: tuple[PlayerPriceResult, ...]


def _to_decimal(
    value: Any,
    field_name: str,
) -> Decimal:
    """Convert one non-negative price into a Decimal value."""

    try:
        decimal_value = Decimal(
            str(value)
        )
    except Exception as error:
        raise ValueError(
            f"{field_name} must be a valid number."
        ) from error

    if not decimal_value.is_finite():
        raise ValueError(
            f"{field_name} must be finite."
        )

    if decimal_value < 0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )

    return decimal_value


def _to_signed_decimal(
    value: Any,
    field_name: str,
) -> Decimal:
    """
    Convert a signed financial change into Decimal.

    Financial changes may be positive, zero or negative.
    """

    try:
        decimal_value = Decimal(
            str(value)
        )
    except Exception as error:
        raise ValueError(
            f"{field_name} must be a valid number."
        ) from error

    if not decimal_value.is_finite():
        raise ValueError(
            f"{field_name} must be finite."
        )

    return decimal_value


def _floor_to_tenth(
    value: Decimal,
) -> Decimal:
    """Round a positive monetary value down to £0.1m."""

    tenths = (
        value
        / FPL_PRICE_INCREMENT
    ).to_integral_value(
        rounding=ROUND_FLOOR
    )

    return (
        tenths
        * FPL_PRICE_INCREMENT
    )


def _money_float(
    value: Decimal,
) -> float:
    """Convert a Decimal price into a one-decimal float."""

    return float(
        value.quantize(
            FPL_PRICE_INCREMENT
        )
    )


def calculate_selling_price(
    purchase_price: float,
    current_price: float,
) -> float:
    """
    Calculate an owned player's official FPL selling price.

    Price falls are absorbed in full.

    When the current price exceeds the purchase price, the
    manager keeps half of the increase, rounded down to the
    nearest £0.1m.
    """

    purchase = _to_decimal(
        purchase_price,
        "purchase_price",
    )

    current = _to_decimal(
        current_price,
        "current_price",
    )

    if current <= purchase:
        return _money_float(
            current
        )

    price_increase = (
        current
        - purchase
    )

    retained_profit = _floor_to_tenth(
        price_increase
        / Decimal("2")
    )

    selling_price = (
        purchase
        + retained_profit
    )

    return _money_float(
        selling_price
    )


def evaluate_player_price(
    player_id: int,
    player_name: str,
    purchase_price: float,
    current_price: float,
) -> PlayerPriceResult:
    """Calculate a full price breakdown for one owned player."""

    parsed_player_id = int(
        player_id
    )

    if parsed_player_id <= 0:
        raise ValueError(
            "player_id must be positive."
        )

    purchase = _to_decimal(
        purchase_price,
        "purchase_price",
    )

    current = _to_decimal(
        current_price,
        "current_price",
    )

    selling = _to_decimal(
        calculate_selling_price(
            purchase_price=float(purchase),
            current_price=float(current),
        ),
        "selling_price",
    )

    market_change = (
        current
        - purchase
    )

    realised_change = (
        selling
        - purchase
    )

    unrealised_profit = max(
        market_change,
        Decimal("0.0"),
    )

    retained_profit = max(
        realised_change,
        Decimal("0.0"),
    )

    lost_value = max(
        purchase
        - selling,
        Decimal("0.0"),
    )

    return PlayerPriceResult(
        player_id=parsed_player_id,
        player_name=str(
            player_name
        ),
        purchase_price=_money_float(
            purchase
        ),
        current_price=_money_float(
            current
        ),
        selling_price=_money_float(
            selling
        ),
        market_change=_money_float(
            market_change
        ),
        realised_change=_money_float(
            realised_change
        ),
        unrealised_profit=_money_float(
            unrealised_profit
        ),
        retained_profit=_money_float(
            retained_profit
        ),
        lost_value=_money_float(
            lost_value
        ),
    )


def add_selling_prices(
    squad: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add FPL selling-price columns to an owned squad DataFrame.

    Required columns:
    - id
    - player_name
    - purchase_price
    - price
    """

    required_columns = {
        "id",
        "player_name",
        "purchase_price",
        "price",
    }

    missing_columns = required_columns.difference(
        squad.columns
    )

    if missing_columns:
        raise ValueError(
            "Squad is missing price-engine columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    priced_squad = squad.copy()

    priced_squad["id"] = pd.to_numeric(
        priced_squad["id"],
        errors="coerce",
    )

    priced_squad["purchase_price"] = pd.to_numeric(
        priced_squad["purchase_price"],
        errors="coerce",
    )

    priced_squad["price"] = pd.to_numeric(
        priced_squad["price"],
        errors="coerce",
    )

    invalid_rows = priced_squad[
        priced_squad[
            [
                "id",
                "purchase_price",
                "price",
            ]
        ].isna().any(axis=1)
    ]

    if not invalid_rows.empty:
        raise ValueError(
            "Squad contains invalid player IDs or prices."
        )

    price_results = [
        evaluate_player_price(
            player_id=int(
                player["id"]
            ),
            player_name=str(
                player["player_name"]
            ),
            purchase_price=float(
                player["purchase_price"]
            ),
            current_price=float(
                player["price"]
            ),
        )
        for _, player in priced_squad.iterrows()
    ]

    results_by_id = {
        result.player_id: result
        for result in price_results
    }

    priced_squad["selling_price"] = (
        priced_squad["id"]
        .astype(int)
        .map(
            lambda player_id: (
                results_by_id[
                    player_id
                ].selling_price
            )
        )
        .astype(float)
    )

    priced_squad["market_price_change"] = (
        priced_squad["id"]
        .astype(int)
        .map(
            lambda player_id: (
                results_by_id[
                    player_id
                ].market_change
            )
        )
        .astype(float)
    )

    priced_squad["realised_price_change"] = (
        priced_squad["id"]
        .astype(int)
        .map(
            lambda player_id: (
                results_by_id[
                    player_id
                ].realised_change
            )
        )
        .astype(float)
    )

    priced_squad["retained_profit"] = (
        priced_squad["id"]
        .astype(int)
        .map(
            lambda player_id: (
                results_by_id[
                    player_id
                ].retained_profit
            )
        )
        .astype(float)
    )

    priced_squad["lost_value"] = (
        priced_squad["id"]
        .astype(int)
        .map(
            lambda player_id: (
                results_by_id[
                    player_id
                ].lost_value
            )
        )
        .astype(float)
    )

    return priced_squad


def calculate_squad_value(
    squad: pd.DataFrame,
    bank: float,
) -> SquadValueResult:
    """Calculate the full financial value of an owned squad."""

    priced_squad = add_selling_prices(
        squad
    )

    bank_decimal = _to_decimal(
        bank,
        "bank",
    )

    player_prices = tuple(
        evaluate_player_price(
            player_id=int(
                player["id"]
            ),
            player_name=str(
                player["player_name"]
            ),
            purchase_price=float(
                player["purchase_price"]
            ),
            current_price=float(
                player["price"]
            ),
        )
        for _, player in priced_squad.iterrows()
    )

    purchase_value = sum(
        (
            _to_decimal(
                result.purchase_price,
                "purchase_price",
            )
            for result in player_prices
        ),
        Decimal("0.0"),
    )

    current_market_value = sum(
        (
            _to_decimal(
                result.current_price,
                "current_price",
            )
            for result in player_prices
        ),
        Decimal("0.0"),
    )

    selling_value = sum(
        (
            _to_decimal(
                result.selling_price,
                "selling_price",
            )
            for result in player_prices
        ),
        Decimal("0.0"),
    )

    market_change = sum(
        (
            _to_signed_decimal(
                result.market_change,
                "market_change",
            )
            for result in player_prices
        ),
        Decimal("0.0"),
    )

    realised_change = sum(
        (
            _to_signed_decimal(
                result.realised_change,
                "realised_change",
            )
            for result in player_prices
        ),
        Decimal("0.0"),
    )

    unrealised_profit = sum(
        (
            _to_decimal(
                result.unrealised_profit,
                "unrealised_profit",
            )
            for result in player_prices
        ),
        Decimal("0.0"),
    )

    retained_profit = sum(
        (
            _to_decimal(
                result.retained_profit,
                "retained_profit",
            )
            for result in player_prices
        ),
        Decimal("0.0"),
    )

    lost_value = sum(
        (
            _to_decimal(
                result.lost_value,
                "lost_value",
            )
            for result in player_prices
        ),
        Decimal("0.0"),
    )

    available_budget = (
        selling_value
        + bank_decimal
    )

    return SquadValueResult(
        purchase_value=_money_float(
            purchase_value
        ),
        current_market_value=_money_float(
            current_market_value
        ),
        selling_value=_money_float(
            selling_value
        ),
        bank=_money_float(
            bank_decimal
        ),
        available_budget=_money_float(
            available_budget
        ),
        market_change=_money_float(
            market_change
        ),
        realised_change=_money_float(
            realised_change
        ),
        unrealised_profit=_money_float(
            unrealised_profit
        ),
        retained_profit=_money_float(
            retained_profit
        ),
        lost_value=_money_float(
            lost_value
        ),
        player_prices=player_prices,
    )


def calculate_transfer_bank_after(
    bank_before: float,
    outgoing_selling_prices: list[float],
    incoming_purchase_prices: list[float],
) -> float:
    """
    Calculate the bank remaining after a transfer group.

    All outgoing values must already be official selling prices.
    Incoming players are purchased at their current market prices.
    """

    if len(outgoing_selling_prices) != len(
        incoming_purchase_prices
    ):
        raise ValueError(
            "Outgoing and incoming transfer lists "
            "must contain the same number of players."
        )

    bank = _to_decimal(
        bank_before,
        "bank_before",
    )

    outgoing_total = sum(
        (
            _to_decimal(
                price,
                "outgoing_selling_price",
            )
            for price in outgoing_selling_prices
        ),
        Decimal("0.0"),
    )

    incoming_total = sum(
        (
            _to_decimal(
                price,
                "incoming_purchase_price",
            )
            for price in incoming_purchase_prices
        ),
        Decimal("0.0"),
    )

    bank_after = (
        bank
        + outgoing_total
        - incoming_total
    )

    if bank_after < 0:
        raise ValueError(
            "Transfer plan is unaffordable by "
            f"£{abs(_money_float(bank_after)):.1f}m."
        )

    return _money_float(
        bank_after
    )


def transfer_plan_is_affordable(
    bank_before: float,
    outgoing_selling_prices: list[float],
    incoming_purchase_prices: list[float],
) -> bool:
    """Return whether a proposed transfer group is affordable."""

    try:
        calculate_transfer_bank_after(
            bank_before=bank_before,
            outgoing_selling_prices=(
                outgoing_selling_prices
            ),
            incoming_purchase_prices=(
                incoming_purchase_prices
            ),
        )
    except ValueError:
        return False

    return True