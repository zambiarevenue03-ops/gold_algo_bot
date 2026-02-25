# backtest/signal_frequency.py

import pandas as pd
from strategy.entry_logic import check_entry_signal


MIN_GAP_HOURS = 8     # minimum candles between signals
STEP = 5             # speed control


def is_london_ny_overlap(ts):
    """
    London–New York overlap (UTC)
    """
    return 7 <= ts.hour <= 17


def analyze_signal_frequency(df_1h: pd.DataFrame) -> pd.DataFrame:
    signals = []

    print("Starting signal scan...")

    for i in range(200, len(df_1h), STEP):
        slice_1h = df_1h.iloc[:i]
        current_time = slice_1h.index[-1]

        # ----------------------------------
        # SESSION FILTER
        # ----------------------------------
        if not is_london_ny_overlap(current_time):
            continue

        # ----------------------------------
        # SIGNAL SPACING FILTER
        # ----------------------------------
        if signals:
            last_signal_time = signals[-1]["timestamp"]
            hours_since_last = (
                current_time - last_signal_time
            ).total_seconds() / 3600

            if hours_since_last < MIN_GAP_HOURS:
                continue

        # ----------------------------------
        # BUILD HTF DATA
        # ----------------------------------
        df_4h = (
            slice_1h
            .resample("4h")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
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
                "volume": "sum"
            })
            .dropna()
        )

        # ----------------------------------
        # ENTRY SIGNAL
        # ----------------------------------
        signal = check_entry_signal(
            df_1h=slice_1h,
            df_4h=df_4h,
            df_1d=df_1d,
            atr_period=14,
            atr_ma_period=14,
            obv_ma_period=10,
            pivot_lookback=2
        )

        if signal:
            signals.append({
                "timestamp": current_time,
                "direction": signal
            })

    print("Signal scan complete.")

    return pd.DataFrame(signals)
