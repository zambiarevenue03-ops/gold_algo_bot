import pandas as pd
import numpy as np
from typing import Optional


# -- helpers --------------------------------------------------------------------

def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Inline ATR to keep this module self-contained."""
    hi_lo = df["high"] - df["low"]
    hi_pc = (df["high"] - df["close"].shift(1)).abs()
    lo_pc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hi_lo, hi_pc, lo_pc], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


# -- core detection -------------------------------------------------------------

def find_fvgs(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 100,
    min_gap_atr_ratio: float = 0.1,
    atr_period: int = 14,
) -> list:
    """
    Scan the last `lookback` candles of `df` for Fair Value Gaps.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex. Must have at least 3 rows.
    direction : str
        "bullish" → look for bullish FVGs (buy-side imbalances).
        "bearish" → look for bearish FVGs (sell-side imbalances).
    lookback : int
        How many candles back to scan (from the most recent candle).
        Default 100 keeps the search fast on 15M data.
    min_gap_atr_ratio : float
        Minimum gap size as a fraction of ATR.
        Filters out tiny, insignificant imbalances.
        Default 0.1 = gap must be at least 10% of ATR.
    atr_period : int
        ATR period used for the minimum gap filter.

    Returns
    -------
    list of dict, most recent FVG first. Each dict contains:
        {
          "type":        "bullish" | "bearish",
          "fvg_high":    float,   # top of the gap zone
          "fvg_low":     float,   # bottom of the gap zone
          "gap_size":    float,   # fvg_high - fvg_low
          "midpoint":    float,   # midpoint of the gap
          "candle_index": int,    # integer iloc of the impulse candle (candle i-1)
          "timestamp":   pd.Timestamp,  # time of the impulse candle
        }
    """
    if df is None or len(df) < 3:
        return []
    if direction not in ("bullish", "bearish"):
        raise ValueError(f"direction must be 'bullish' or 'bearish', got '{direction}'")

    atr = _calculate_atr(df, atr_period)
    start = max(3, len(df) - lookback)
    fvgs = []

    for i in range(start, len(df)):
        # Three-candle window: prev2, prev1 (impulse), current
        prev2 = i - 2
        # impulse_idx = i - 1  (the candle that caused the gap)
        curr  = i

        prev2_high  = df["high"].iloc[prev2]
        prev2_low   = df["low"].iloc[prev2]
        curr_high   = df["high"].iloc[curr]
        curr_low    = df["low"].iloc[curr]
        current_atr = atr.iloc[curr]

        if direction == "bullish":
            # Gap: high of candle[i-2] must be BELOW low of candle[i]
            if prev2_high < curr_low:
                gap_size = curr_low - prev2_high
                # Filter: gap must be meaningful (not noise)
                if gap_size >= min_gap_atr_ratio * current_atr:
                    fvgs.append({
                        "type":         "bullish",
                        "fvg_high":     curr_low,       # top of gap = bottom of candle[i]
                        "fvg_low":      prev2_high,     # bottom of gap = top of candle[i-2]
                        "gap_size":     gap_size,
                        "midpoint":     (curr_low + prev2_high) / 2,
                        "candle_index": i - 1,
                        "timestamp":    df.index[i - 1],
                    })

        else:  # bearish
            # Gap: low of candle[i-2] must be ABOVE high of candle[i]
            if prev2_low > curr_high:
                gap_size = prev2_low - curr_high
                if gap_size >= min_gap_atr_ratio * current_atr:
                    fvgs.append({
                        "type":         "bearish",
                        "fvg_high":     prev2_low,      # top of gap = bottom of candle[i-2]
                        "fvg_low":      curr_high,      # bottom of gap = top of candle[i]
                        "gap_size":     gap_size,
                        "midpoint":     (prev2_low + curr_high) / 2,
                        "candle_index": i - 1,
                        "timestamp":    df.index[i - 1],
                    })

    # Return most recent first
    return list(reversed(fvgs))


def get_latest_fvg(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 100,
    min_gap_atr_ratio: float = 0.1,
    atr_period: int = 14,
    exclude_mitigated: bool = True,
) -> Optional[dict]:
    """
    Return the single most recent unmitigated FVG, or None.

    Parameters
    ----------
    exclude_mitigated : bool
        If True (default), skip FVGs that price has already fully filled.
        An FVG is considered mitigated when:
          - bullish: a subsequent candle's low trades BELOW fvg_low
          - bearish: a subsequent candle's high trades ABOVE fvg_high
    """
    fvgs = find_fvgs(df, direction, lookback, min_gap_atr_ratio, atr_period)
    if not fvgs:
        return None

    if not exclude_mitigated:
        return fvgs[0]

    for fvg in fvgs:
        if not fvg_has_been_mitigated(df, fvg):
            return fvg

    return None


# -- zone queries ---------------------------------------------------------------

def price_in_fvg(price: float, fvg: dict) -> bool:
    """
    Returns True if `price` is currently inside the FVG zone.

    For a bullish FVG this means price has retraced DOWN into the gap
    (a potential entry zone for a long).

    For a bearish FVG this means price has retraced UP into the gap
    (a potential entry zone for a short).
    """
    if fvg is None:
        return False
    return fvg["fvg_low"] <= price <= fvg["fvg_high"]


def fvg_has_been_mitigated(df: pd.DataFrame, fvg: dict) -> bool:
    """
    Returns True if the FVG has already been fully mitigated (filled).

    Mitigation rules:
        Bullish FVG: a candle AFTER the FVG trades BELOW fvg_low
                     (gap has been completely filled - no longer valid)
        Bearish FVG: a candle AFTER the FVG trades ABOVE fvg_high

    Note: partial mitigation (price enters but doesn't fully fill) still
    leaves the FVG valid for entry at its boundary.
    """
    if fvg is None:
        return False

    impulse_idx = fvg["candle_index"]
    if impulse_idx >= len(df) - 1:
        # FVG is the last candle - nothing after it yet
        return False

    future = df.iloc[impulse_idx + 1:]

    if fvg["type"] == "bullish":
        # Mitigated if any future candle's low goes below the bottom of the gap
        return bool((future["low"] < fvg["fvg_low"]).any())
    else:
        # Mitigated if any future candle's high goes above the top of the gap
        return bool((future["high"] > fvg["fvg_high"]).any())


def price_approaching_fvg(
    price: float,
    fvg: dict,
    approach_buffer_pct: float = 0.002,
) -> bool:
    """
    Returns True if price is approaching (but not yet inside) the FVG zone.
    Useful for pre-positioning limit orders slightly above/below the zone.

    approach_buffer_pct: how close price needs to be as % of price.
    Default 0.2% = within $5 on gold at $2500.
    """
    if fvg is None:
        return False
    buffer = price * approach_buffer_pct
    if fvg["type"] == "bullish":
        # Price is above fvg_high and falling toward the gap (within buffer)
        return fvg["fvg_high"] <= price <= fvg["fvg_high"] + buffer
    else:
        # Price is below fvg_low and rising toward the gap (within buffer)
        return fvg["fvg_low"] - buffer <= price <= fvg["fvg_low"]


# -- summary --------------------------------------------------------------------

def summarise_fvgs(fvgs: list) -> None:
    """Print a human-readable summary of detected FVGs. Used for debugging."""
    if not fvgs:
        print("  No FVGs found.")
        return
    print(f"  Found {len(fvgs)} FVG(s):")
    for i, fvg in enumerate(fvgs[:5]):  # show max 5
        print(
            f"    [{i+1}] {fvg['type'].upper():8s} | "
            f"{fvg['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
            f"Zone: {fvg['fvg_low']:.2f} - {fvg['fvg_high']:.2f} | "
            f"Gap: {fvg['gap_size']:.2f}"
        )