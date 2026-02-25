# strategy/entry_logic.py

import pandas as pd
from typing import Optional

from indicators.atr import calculate_atr, atr_moving_average
from indicators.obv import calculate_obv, obv_moving_average
from indicators.pivots import detect_swings

from smc.structure import determine_trend
from smc.choch import detect_choch
from smc.liquidity import (
    find_equal_highs,
    find_equal_lows,
    detect_liquidity_sweep
)
from smc.order_blocks import find_order_block, price_in_order_block


def check_entry_signal(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    atr_period: int = 14,
    atr_ma_period: int = 14,
    obv_ma_period: int = 10,
    pivot_lookback: int = 2
) -> Optional[str]:

    # ==================================================
    # 1️⃣ HTF STRUCTURE (HARD FILTER)
    # ==================================================
    sh_1d, sl_1d = detect_swings(df_1d, pivot_lookback)
    sh_4h, sl_4h = detect_swings(df_4h, pivot_lookback)

    trend_1d = determine_trend(sh_1d, sl_1d)
    trend_4h = determine_trend(sh_4h, sl_4h)

    if trend_1d is None or trend_1d != trend_4h:
        return None

    bias = trend_4h  # 'bullish' or 'bearish'

    # ==================================================
    # 2️⃣ LIQUIDITY SWEEP (HARD FILTER)
    # ==================================================
    sh_1h, sl_1h = detect_swings(df_1h, pivot_lookback)

    liquidity_swept = False

    if bias == "bullish":
        eq_lows = find_equal_lows(df_1h, sl_1h)
        if eq_lows:
            level, _ = eq_lows
            liquidity_swept = detect_liquidity_sweep(
                df_1h, level, side="sell"
            )

    elif bias == "bearish":
        eq_highs = find_equal_highs(df_1h, sh_1h)
        if eq_highs:
            level, _ = eq_highs
            liquidity_swept = detect_liquidity_sweep(
                df_1h, level, side="buy"
            )

    if not liquidity_swept:
        return None

    # ==================================================
    # 3️⃣ ORDER BLOCK (HARD FILTER)
    # ==================================================
    ob = find_order_block(df_4h, direction=bias)
    if not ob:
        return None

    current_price = df_1h["close"].iloc[-1]

    if not price_in_order_block(
        current_price,
        ob["zone_low"],
        ob["zone_high"]
    ):
        return None

    # ==================================================
    # 4️⃣ SOFT CONFIRMATIONS (SCORING)
    # ==================================================
    score = 0

    # ---- OBV (SOFT)
    obv_4h = calculate_obv(df_4h)
    obv_ma_4h = obv_moving_average(obv_4h, obv_ma_period)

    if bias == "bullish" and obv_4h.iloc[-1] > obv_ma_4h.iloc[-1]:
        score += 1
    elif bias == "bearish" and obv_4h.iloc[-1] < obv_ma_4h.iloc[-1]:
        score += 1

    # ---- ATR expansion (SOFT, relaxed)
    atr_4h = calculate_atr(df_4h, atr_period)
    atr_ma_4h = atr_moving_average(atr_4h, atr_ma_period)

    if atr_4h.iloc[-1] > atr_ma_4h.iloc[-1]:
        score += 1

    # ---- CHoCH (SOFT)
    choch = detect_choch(
        df_1h,
        sh_1h,
        sl_1h,
        current_trend=bias
    )

    if choch:
        score += 1

    # ==================================================
    # 5️⃣ FINAL DECISION
    # ==================================================
    if score == 0:
        return None

    return "long" if bias == "bullish" else "short"
