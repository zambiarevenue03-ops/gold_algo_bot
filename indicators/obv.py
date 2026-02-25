# indicators/obv.py

import pandas as pd
import numpy as np


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV)

    Required columns: ['close', 'volume']
    """
    direction = np.sign(df['close'].diff()).fillna(0)
    obv = (direction * df['volume']).cumsum()

    return obv


def obv_moving_average(obv: pd.Series, period: int = 10) -> pd.Series:
    """Smooth OBV with moving average"""
    return obv.rolling(window=period).mean()


def obv_direction(obv: pd.Series, obv_ma: pd.Series) -> pd.Series:
    """
    Returns:
    +1 → bullish volume
    -1 → bearish volume
     0 → neutral
    """
    direction = pd.Series(index=obv.index, dtype=int)

    direction[obv > obv_ma] = 1
    direction[obv < obv_ma] = -1
    direction[(obv - obv_ma).abs() < 1e-6] = 0

    return direction
