# smc/choch.py

import pandas as pd
from typing import List


def detect_choch(
    df: pd.DataFrame,
    swing_highs: List[int],
    swing_lows: List[int],
    current_trend: str
) -> bool:
    """
    Detect Change of Character (CHoCH)

    Required columns: ['close']

    Returns:
        True  → CHoCH detected
        False → no CHoCH
    """

    if current_trend not in ('bullish', 'bearish'):
        return False

    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return False

    last_close = df['close'].iloc[-1]

    # Bullish → Bearish CHoCH
    if current_trend == 'bullish':
        last_higher_low_idx = swing_lows[-1]
        last_higher_low_price = df['low'].iloc[last_higher_low_idx]

        if last_close < last_higher_low_price:
            return True

    # Bearish → Bullish CHoCH
    if current_trend == 'bearish':
        last_lower_high_idx = swing_highs[-1]
        last_lower_high_price = df['high'].iloc[last_lower_high_idx]

        if last_close > last_lower_high_price:
            return True

    return False
