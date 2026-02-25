# risk/trade_levels.py

from typing import Dict, Optional
import pandas as pd


def calculate_stop_loss(
    *,
    df_1h: pd.DataFrame,
    bias: str,
    order_block: Dict[str, float],
    liquidity_level: float
) -> Optional[Dict[str, float]]:
    """
    Calculates a structural stop loss based on:
    - Liquidity sweep level
    - Higher-timeframe order block

    Parameters
    ----------
    df_1h : pd.DataFrame
        1H dataframe (must contain 'close')
    bias : str
        'bullish' or 'bearish'
    order_block : dict
        Must contain:
            - 'zone_low'
            - 'zone_high'
    liquidity_level : float
        Price level swept (equal high / low)

    Returns
    -------
    dict or None
        {
            "stop_loss": float,
            "risk": float
        }
    """

    if df_1h.empty:
        return None

    entry_price = df_1h["close"].iloc[-1]

    # -------------------------------
    # LONG TRADE STOP LOSS
    # -------------------------------
    if bias == "bullish":
        stop_loss = min(
            liquidity_level,
            order_block["zone_low"]
        )

        risk = entry_price - stop_loss

        if risk <= 0:
            return None

    # -------------------------------
    # SHORT TRADE STOP LOSS
    # -------------------------------
    elif bias == "bearish":
        stop_loss = max(
            liquidity_level,
            order_block["zone_high"]
        )

        risk = stop_loss - entry_price

        if risk <= 0:
            return None

    else:
        return None

    return {
        "stop_loss": round(stop_loss, 2),
        "risk": round(risk, 2)
    }
