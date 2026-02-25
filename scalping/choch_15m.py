import pandas as pd
import numpy as np
from typing import Optional
import sys
import os

# Import pivot detection from existing swing bot indicators
# This keeps the logic DRY and ensures consistency across both bots
try:
    from indicators.pivots import detect_swings
except ImportError:
    # Fallback for when running tests standalone
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from indicators.pivots import detect_swings


# ── core detection ─────────────────────────────────────────────────────────────

def detect_choch_15m(
    df: pd.DataFrame,
    direction: str,
    pivot_lookback: int = 2,
    min_break_distance_pct: float = 0.0005,
    volume_confirm: bool = False,
    volume_spike_threshold: float = 1.3,
) -> Optional[dict]:
    """
    Detect a Change of Character on the 15M timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        15M OHLCV DataFrame with DatetimeIndex.
        Must have at least 20+ candles for reliable swing detection.
    direction : str
        Expected CHoCH direction:
          "bullish" → detect bullish CHoCH (break ABOVE recent lower high)
          "bearish" → detect bearish CHoCH (break BELOW recent higher low)
    pivot_lookback : int
        X parameter for swing detection (see indicators/pivots.py).
        Default 2 — suitable for 15M noise levels.
        Larger values (3-5) require more confirmation but reduce false positives.
    min_break_distance_pct : float
        Minimum percentage break beyond the swing level to avoid false triggers.
        Default 0.05% (~$1.25 on gold at $2500).
        Set to 0.0 to disable (not recommended on 15M).
    volume_confirm : bool
        If True, require a volume spike on the CHoCH candle.
        Default False — volume is unreliable on many brokers' 15M data.
    volume_spike_threshold : float
        Volume must be >= this multiple of 20-period average volume.
        Only applies if volume_confirm=True.

    Returns
    -------
    dict or None
        If CHoCH detected, returns:
        {
          "direction":      "bullish" | "bearish",
          "timestamp":      pd.Timestamp,        # when CHoCH occurred
          "candle_index":   int,                 # iloc of CHoCH candle
          "broken_level":   float,               # the swing high/low that was broken
          "close_price":    float,               # close of CHoCH candle
          "break_distance": float,               # abs(close - broken_level)
          "break_pct":      float,               # break_distance / broken_level
          "volume":         float,               # volume on CHoCH candle
          "volume_ratio":   float,               # volume / avg_volume (if volume_confirm)
        }
        If no CHoCH detected, returns None.
    """
    if df is None or len(df) < pivot_lookback * 4:
        return None
    if direction not in ("bullish", "bearish"):
        raise ValueError(f"direction must be 'bullish' or 'bearish', got '{direction}'")

    # 1. Detect swing highs and lows using existing pivot logic
    swing_highs, swing_lows = detect_swings(df, lookback=pivot_lookback)

    # 2. Get the most recent swing based on direction
    if direction == "bullish":
        # Bullish CHoCH = break ABOVE the most recent lower high (bearish swing high)
        # This confirms the bearish structure is weakening
        if not swing_highs:
            return None
        broken_swing_idx = swing_highs[-1]
        broken_level = df["high"].iloc[broken_swing_idx]

        # Look for a candle that CLOSES above this level
        future_candles = df.iloc[broken_swing_idx + 1:]
        if len(future_candles) == 0:
            return None

        # Find first candle that breaks above with enough conviction
        for i, (idx, candle) in enumerate(future_candles.iterrows()):
            close = candle["close"]
            break_distance = close - broken_level

            if break_distance >= 0:  # close is above or at the level
                break_pct = abs(break_distance) / broken_level if broken_level > 0 else 0.0

                # Check minimum break distance
                if break_pct >= min_break_distance_pct:
                    choch_candle_iloc = broken_swing_idx + 1 + i

                    result = {
                        "direction":      "bullish",
                        "timestamp":      df.index[choch_candle_iloc],
                        "candle_index":   choch_candle_iloc,
                        "broken_level":   float(broken_level),
                        "close_price":    float(close),
                        "break_distance": float(break_distance),
                        "break_pct":      float(break_pct),
                        "volume":         float(candle["volume"]),
                    }

                    # Volume confirmation (optional)
                    if volume_confirm:
                        avg_vol = df["volume"].iloc[max(0, choch_candle_iloc - 20):choch_candle_iloc].mean()
                        vol_ratio = candle["volume"] / avg_vol if avg_vol > 0 else 0.0
                        result["volume_ratio"] = float(vol_ratio)

                        if vol_ratio < volume_spike_threshold:
                            continue  # Not enough volume — keep searching

                    return result

    else:  # bearish
        # Bearish CHoCH = break BELOW the most recent higher low (bullish swing low)
        if not swing_lows:
            return None
        broken_swing_idx = swing_lows[-1]
        broken_level = df["low"].iloc[broken_swing_idx]

        future_candles = df.iloc[broken_swing_idx + 1:]
        if len(future_candles) == 0:
            return None

        for i, (idx, candle) in enumerate(future_candles.iterrows()):
            close = candle["close"]
            break_distance = broken_level - close  # positive when break below

            if break_distance >= 0:  # close is below or at the level
                break_pct = abs(break_distance) / broken_level if broken_level > 0 else 0.0

                if break_pct >= min_break_distance_pct:
                    choch_candle_iloc = broken_swing_idx + 1 + i

                    result = {
                        "direction":      "bearish",
                        "timestamp":      df.index[choch_candle_iloc],
                        "candle_index":   choch_candle_iloc,
                        "broken_level":   float(broken_level),
                        "close_price":    float(close),
                        "break_distance": float(break_distance),
                        "break_pct":      float(break_pct),
                        "volume":         float(candle["volume"]),
                    }

                    if volume_confirm:
                        avg_vol = df["volume"].iloc[max(0, choch_candle_iloc - 20):choch_candle_iloc].mean()
                        vol_ratio = candle["volume"] / avg_vol if avg_vol > 0 else 0.0
                        result["volume_ratio"] = float(vol_ratio)

                        if vol_ratio < volume_spike_threshold:
                            continue

                    return result

    return None


# ── convenience queries ────────────────────────────────────────────────────────

def recent_choch_exists(
    df: pd.DataFrame,
    direction: str,
    max_candles_ago: int = 10,
    **kwargs,
) -> bool:
    """
    Returns True if a CHoCH in the given direction occurred within the
    last `max_candles_ago` candles.

    Useful for filtering entries: only enter if CHoCH happened recently
    (not 50 candles ago when the structure has changed again).
    """
    choch = detect_choch_15m(df, direction, **kwargs)
    if choch is None:
        return False

    current_idx = len(df) - 1
    candles_ago = current_idx - choch["candle_index"]

    return candles_ago <= max_candles_ago


def choch_aligns_with_bias(
    df: pd.DataFrame,
    htf_bias: str,
    max_candles_ago: int = 10,
    **kwargs,
) -> bool:
    """
    Returns True if there is a recent CHoCH that aligns with the HTF bias.

    This is the key confirmation for scalp entries:
      - HTF bias is bullish → look for bullish CHoCH on 15M
      - HTF bias is bearish → look for bearish CHoCH on 15M

    If HTF says "long" but 15M shows a bearish CHoCH, this returns False.
    """
    if htf_bias not in ("bullish", "bearish"):
        return False

    return recent_choch_exists(df, htf_bias, max_candles_ago, **kwargs)


def get_choch_age_candles(df: pd.DataFrame, direction: str, **kwargs) -> int:
    """
    Returns how many candles ago the CHoCH occurred.
    Returns -1 if no CHoCH detected.

    Useful for filtering stale setups.
    """
    choch = detect_choch_15m(df, direction, **kwargs)
    if choch is None:
        return -1

    current_idx = len(df) - 1
    return current_idx - choch["candle_index"]


# ── debugging helpers ──────────────────────────────────────────────────────────

def summarise_choch(choch: Optional[dict]) -> None:
    """Print a human-readable CHoCH summary. Used for debugging."""
    if choch is None:
        print("  No CHoCH detected.")
        return

    print(f"  CHoCH detected:")
    print(f"    Direction:      {choch['direction'].upper()}")
    print(f"    Timestamp:      {choch['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    print(f"    Broken level:   {choch['broken_level']:.2f}")
    print(f"    Close price:    {choch['close_price']:.2f}")
    print(f"    Break distance: {choch['break_distance']:.2f} ({choch['break_pct']*100:.3f}%)")
    print(f"    Volume:         {choch['volume']:.0f}")
    if "volume_ratio" in choch:
        print(f"    Volume ratio:   {choch['volume_ratio']:.2f}×")
