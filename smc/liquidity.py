# smc/liquidity.py

import pandas as pd
from typing import List, Tuple, Optional


def find_equal_highs(
    df: pd.DataFrame,
    swing_highs: List[int],
    tolerance: float = 0.001,
    lookback: int = 5
) -> Optional[Tuple[float, List[int]]]:
    """
    Detect equal highs (buy-side liquidity)

    tolerance: percentage difference allowed (e.g. 0.001 = 0.1%)
    """

    if len(swing_highs) < 2:
        return None

    recent_highs = swing_highs[-lookback:]
    prices = [df['high'].iloc[i] for i in recent_highs]

    base_price = prices[-1]
    equal_indices = []

    for idx, price in zip(recent_highs, prices):
        if abs(price - base_price) / base_price <= tolerance:
            equal_indices.append(idx)

    if len(equal_indices) >= 2:
        return base_price, equal_indices

    return None


def find_equal_lows(
    df: pd.DataFrame,
    swing_lows: List[int],
    tolerance: float = 0.001,
    lookback: int = 5
) -> Optional[Tuple[float, List[int]]]:
    """
    Detect equal lows (sell-side liquidity)
    """

    if len(swing_lows) < 2:
        return None

    recent_lows = swing_lows[-lookback:]
    prices = [df['low'].iloc[i] for i in recent_lows]

    base_price = prices[-1]
    equal_indices = []

    for idx, price in zip(recent_lows, prices):
        if abs(price - base_price) / base_price <= tolerance:
            equal_indices.append(idx)

    if len(equal_indices) >= 2:
        return base_price, equal_indices

    return None


def detect_liquidity_sweep(
    df: pd.DataFrame,
    level_price: float,
    side: str
) -> bool:
    """
    Detect a liquidity sweep on the latest candle.

    side:
      'buy'  → sweep above equal highs
      'sell' → sweep below equal lows
    """

    last_candle = df.iloc[-1]

    if side == 'buy':
        # Wick above, close below
        if last_candle['high'] > level_price and last_candle['close'] < level_price:
            return True

    if side == 'sell':
        # Wick below, close above
        if last_candle['low'] < level_price and last_candle['close'] > level_price:
            return True

    return False
