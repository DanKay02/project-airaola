import pandas as pd
import pytest

from airaola.finance.price_engine import (
    add_selling_prices,
    calculate_selling_price,
    calculate_squad_value,
    calculate_transfer_bank_after,
    transfer_plan_is_affordable,
)


@pytest.mark.parametrize(
    (
        "purchase_price",
        "current_price",
        "expected_selling_price",
    ),
    [
        (7.5, 7.5, 7.5),
        (7.5, 7.6, 7.5),
        (7.5, 7.7, 7.6),
        (7.5, 7.8, 7.6),
        (7.5, 7.9, 7.7),
        (7.5, 8.0, 7.7),
        (7.5, 7.4, 7.4),
        (7.5, 7.0, 7.0),
    ],
)
def test_calculate_selling_price(
    purchase_price: float,
    current_price: float,
    expected_selling_price: float,
) -> None:
    assert calculate_selling_price(
        purchase_price=purchase_price,
        current_price=current_price,
    ) == expected_selling_price


def test_add_selling_prices() -> None:
    squad = pd.DataFrame(
        [
            {
                "id": 1,
                "player_name": "Profit Player",
                "purchase_price": 7.5,
                "price": 7.8,
            },
            {
                "id": 2,
                "player_name": "Falling Player",
                "purchase_price": 6.0,
                "price": 5.7,
            },
        ]
    )

    result = add_selling_prices(
        squad
    )

    assert result.loc[
        result["id"] == 1,
        "selling_price",
    ].iloc[0] == 7.6

    assert result.loc[
        result["id"] == 2,
        "selling_price",
    ].iloc[0] == 5.7


def test_calculate_squad_value() -> None:
    squad = pd.DataFrame(
        [
            {
                "id": 1,
                "player_name": "Profit Player",
                "purchase_price": 7.5,
                "price": 7.8,
            },
            {
                "id": 2,
                "player_name": "Falling Player",
                "purchase_price": 6.0,
                "price": 5.7,
            },
        ]
    )

    result = calculate_squad_value(
        squad=squad,
        bank=1.0,
    )

    assert result.purchase_value == 13.5
    assert result.current_market_value == 13.5
    assert result.selling_value == 13.3
    assert result.available_budget == 14.3
    assert result.retained_profit == 0.1
    assert result.lost_value == 0.3


def test_transfer_bank_after() -> None:
    bank_after = calculate_transfer_bank_after(
        bank_before=0.5,
        outgoing_selling_prices=[
            6.2,
        ],
        incoming_purchase_prices=[
            6.5,
        ],
    )

    assert bank_after == 0.2


def test_unaffordable_transfer_plan() -> None:
    assert not transfer_plan_is_affordable(
        bank_before=0.0,
        outgoing_selling_prices=[
            5.0,
        ],
        incoming_purchase_prices=[
            5.1,
        ],
    )