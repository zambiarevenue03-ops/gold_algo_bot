# backtest/sl_only_backtest.py

import pandas as pd
from typing import List, Dict

from strategy.entry_logic import check_entry_signal
from risk.trade_levels import calculate_stop_loss


def run_sl_only_backtest(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Backtests stop-loss validity only.
    A trade is considered 'failed' if price touches SL at any future point.
    """

    trades: List[Dict] = []

    for i in range(200, len(df_1h)):
        slice_1h = df_1h.iloc[:i]

        # ---------------------------
        # Build HTF data
        # ---------------------------
        df_4h = (
            slice_1h
            .resample("4h")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
        )

        df_1d = (
            slice_1h
            .resample("1d")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
        )

        # ---------------------------
        # ENTRY SIGNAL
        # ---------------------------
        signal = check_entry_signal(
    df_1h=slice_1h,
    df_4h=df_4h,
    df_1d=df_1d,
    atr_period=14,
    atr_ma_period=14,
    obv_ma_period=10,
    pivot_lookback=2,
)

if not signal:
    continue

# infer bias from existing signal
bias = "bullish" if signal == "long" else "bearish"

# TEMP placeholders (until we upgrade entry context)
order_block = None
liquidity_level = None


        # ---------------------------
        # STOP LOSS CALCULATION
        # ---------------------------
recent_low = slice_1h["low"].iloc[-5:].min()
recent_high = slice_1h["high"].iloc[-5:].max()

if bias == "bullish":
    stop_loss = recent_low
else:
    stop_loss = recent_high

entry_price = slice_1h["close"].iloc[-1]
risk = abs(entry_price - stop_loss)

if risk <= 0:
    continue


if not sl_data:
            continue

        entry_price = slice_1h["close"].iloc[-1]
        stop_loss = sl_data["stop_loss"]

        # ---------------------------
        # SL HIT CHECK
        # ---------------------------
        future_prices = df_1h.iloc[i:]

        if bias == "bullish":
            sl_hit = (future_prices["low"] <= stop_loss).any()
        else:
            sl_hit = (future_prices["high"] >= stop_loss).any()

        trades.append({
    "timestamp": slice_1h.index[-1],
    "direction": bias,
    "entry": entry_price,
    "stop_loss": stop_loss,
    "sl_hit": sl_hit,
    "risk": risk,
})


    return pd.DataFrame(trades)
