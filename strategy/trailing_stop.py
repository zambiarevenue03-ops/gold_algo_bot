# strategy/trailing_stop.py

def update_trailing_stop(
    direction: str,
    entry_price: float,
    current_price: float,
    atr_value: float,
    current_stop: float,
    break_even_trigger_r: float = 1.0,
    trail_atr_multiplier: float = 1.5
) -> float:
    """
    ATR-based trailing stop logic.
    Returns updated stop loss (never loosens risk).
    """

    risk = abs(entry_price - current_stop)

    if direction == "long":
        # Move to break-even
        if current_price >= entry_price + risk * break_even_trigger_r:
            new_stop = max(current_stop, entry_price)
        else:
            return current_stop

        # Trail after BE
        trailing_stop = current_price - atr_value * trail_atr_multiplier
        return max(new_stop, trailing_stop)

    if direction == "short":
        if current_price <= entry_price - risk * break_even_trigger_r:
            new_stop = min(current_stop, entry_price)
        else:
            return current_stop

        trailing_stop = current_price + atr_value * trail_atr_multiplier
        return min(new_stop, trailing_stop)

    return current_stop
