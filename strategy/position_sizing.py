# strategy/position_sizing.py

def fixed_risk_position_size(
    account_balance: float,
    risk_percent: float,
    stop_distance: float,
    pip_value: float
) -> float:
    """
    Fixed % risk position sizing.
    """

    risk_amount = account_balance * risk_percent

    if stop_distance <= 0:
        return 0.0

    size = risk_amount / (stop_distance * pip_value)
    return round(size, 2)
