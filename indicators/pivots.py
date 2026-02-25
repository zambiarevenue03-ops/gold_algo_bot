# indicators/pivots.py

import pandas as pd
from typing import List, Tuple


def detect_swings(
    df: pd.DataFrame,
    lookback: int = 2
) -> Tuple[List[int], List[int]]:
    """
    Detect swing highs and swing lows.

    A swing high is a high that is greater than 'lookback' candles
    before and after it.
    A swing low is a low that is lower than 'lookback' candles
    before and after it.

    Required columns: ['high', 'low']

    Returns:
        swing_highs: list of index positions
        swing_lows: list of index positions
    """

    highs = df['high'].values
    lows = df['low'].values

    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        # Swing High
        if (
            highs[i] > max(highs[i - lookback:i]) and
            highs[i] > max(highs[i + 1:i + lookback + 1])
        ):
            swing_highs.append(i)

        # Swing Low
        if (
            lows[i] < min(lows[i - lookback:i]) and
            lows[i] < min(lows[i + 1:i + lookback + 1])
        ):
            swing_lows.append(i)

    return swing_highs, swing_lows


def label_swings(
    df: pd.DataFrame,
    swing_highs: List[int],
    swing_lows: List[int]
) -> pd.DataFrame:
    """
    Label swing points directly on the dataframe.

    Adds columns:
    - swing_high (bool)
    - swing_low (bool)
    """

    df = df.copy()
    df['swing_high'] = False
    df['swing_low'] = False

    df.loc[df.index[swing_highs], 'swing_high'] = True
    df.loc[df.index[swing_lows], 'swing_low'] = True

    return df


def get_last_n_swings(
    swing_points: List[int],
    n: int = 2
) -> List[int]:
    """
    Return the last n swing points.
    """
    if len(swing_points) < n:
        return []
    return swing_points[-n:]
