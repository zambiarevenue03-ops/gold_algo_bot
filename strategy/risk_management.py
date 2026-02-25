# strategy/risk_management.py

from typing import Dict


def calculate_stop_loss(
    entry_price: float,
    atr_value: float,
    direction: str,
    atr_multiplier: float = 2.0
) -> float:
    """
    Calculate ATR-based stop loss.
    """

    if direction == 'long':
        return entry_price - (atr_value * atr_multiplier)

    if direction == 'short':
        return entry_price + (atr_value * atr_multiplier)

    raise ValueError("Direction must be 'long' or 'short'")


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float,
    pip_value: float
) -> float:
    """
    Calculate position size based on fixed risk.

    pip_value:
        value of 1 price unit (or pip) per 1 lot/contract
    """

    risk_amount = account_balance * risk_percent
    stop_distance = abs(entry_price - stop_loss_price)

    if stop_distance <= 0:
        return 0.0

    position_size = risk_amount / (stop_distance * pip_value)
    return round(position_size, 2)


def calculate_take_profit(
    entry_price: float,
    stop_loss_price: float,
    direction: str,
    reward_risk_ratio: float = 2.0
) -> float:
    """
    Calculate take profit using R-multiple.
    """

    risk = abs(entry_price - stop_loss_price)

    if direction == 'long':
        return entry_price + (risk * reward_risk_ratio)

    if direction == 'short':
        return entry_price - (risk * reward_risk_ratio)

    raise ValueError("Direction must be 'long' or 'short'")


def build_trade_parameters(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    atr_value: float,
    direction: str,
    pip_value: float,
    atr_multiplier: float = 2.0,
    reward_risk_ratio: float = 2.0
) -> Dict:
    """
    Convenience function that returns everything needed for execution.
    """

    stop_loss = calculate_stop_loss(
        entry_price,
        atr_value,
        direction,
        atr_multiplier
    )

    position_size = calculate_position_size(
        account_balance,
        risk_percent,
        entry_price,
        stop_loss,
        pip_value
    )

    take_profit = calculate_take_profit(
        entry_price,
        stop_loss,
        direction,
        reward_risk_ratio
    )

    return {
        "direction": direction,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_size": position_size
    }
