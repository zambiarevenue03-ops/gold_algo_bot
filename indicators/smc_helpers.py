# indicators/smc_helpers.py

import pandas as pd


def candle_body_size(df: pd.DataFrame) -> pd.Series:
    return (df['close'] - df['open']).abs()


def candle_range(df: pd.DataFrame) -> pd.Series:
    return df['high'] - df['low']


def is_bullish_candle(row) -> bool:
    return row['close'] > row['open']


def is_bearish_candle(row) -> bool:
    return row['close'] < row['open']
