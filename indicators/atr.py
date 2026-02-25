# indicators/atr.py

import pandas as pd
import numpy as np


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)

    Required columns: ['high', 'low', 'close']
    """
    high = df['high']
    low = df['low']
    close = df['close']

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def atr_moving_average(atr: pd.Series, period: int = 14) -> pd.Series:
    """ATR moving average (used for regime detection)"""
    return atr.rolling(window=period).mean()


def atr_slope(atr: pd.Series, lookback: int = 3) -> pd.Series:
    """
    ATR slope to detect expansion/contraction
    Positive slope = volatility increasing
    """
    return atr.diff(lookback)
