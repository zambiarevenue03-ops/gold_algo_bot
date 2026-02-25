# indicators/volatility.py

import pandas as pd


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_multiplier: float = 2.0
):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()

    upper = ma + std_multiplier * std
    lower = ma - std_multiplier * std

    return ma, upper, lower


def is_volatility_expanding(
    atr: pd.Series,
    atr_ma: pd.Series
) -> pd.Series:
    """
    Returns True when volatility is expanding.
    """
    return atr > atr_ma
