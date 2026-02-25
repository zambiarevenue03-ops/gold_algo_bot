import pandas as pd
import numpy as np
from typing import Optional


# ── helpers ────────────────────────────────────────────────────────────────────

def _body_size(candle: pd.Series) -> float:
    """Absolute size of candle body (open to close)."""
    return abs(candle["close"] - candle["open"])


def _candle_range(candle: pd.Series) -> float:
    """Full high-to-low range of the candle."""
    return candle["high"] - candle["low"]


def _is_bullish(candle: pd.Series) -> bool:
    return candle["close"] > candle["open"]


def _is_bearish(candle: pd.Series) -> bool:
    return candle["close"] < candle["open"]


def _upper_wick(candle: pd.Series) -> float:
    return candle["high"] - max(candle["open"], candle["close"])


def _lower_wick(candle: pd.Series) -> float:
    return min(candle["open"], candle["close"]) - candle["low"]


def _rolling_avg_body(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling average of absolute body sizes."""
    bodies = (df["close"] - df["open"]).abs()
    return bodies.rolling(window=period, min_periods=5).mean()


def _rolling_avg_range(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling average of high-low ranges."""
    ranges = df["high"] - df["low"]
    return ranges.rolling(window=period, min_periods=5).mean()


# ── core detection ─────────────────────────────────────────────────────────────

def is_displacement(
    candle: pd.Series,
    avg_body: float,
    avg_range: float,
    direction: str,
    body_atr_ratio: float = 1.5,
    min_body_ratio: float = 0.55,
    min_close_position: float = 0.60,
) -> bool:
    """
    Returns True if a single candle qualifies as a displacement.

    Parameters
    ----------
    candle : pd.Series
        A single OHLCV row with keys: open, high, low, close.
    avg_body : float
        Average body size over the lookback period (pre-calculated).
    avg_range : float
        Average candle range over the lookback period (pre-calculated).
    direction : str
        "bullish" or "bearish" — the expected displacement direction.
    body_atr_ratio : float
        Body must be >= this multiple of avg_body.
        Default 1.5 = body must be 50% bigger than recent average.
    min_body_ratio : float
        Body must be >= this fraction of the candle's total range.
        Default 0.55 = body must be at least 55% of high-to-low range.
        Prevents wick-heavy candles from qualifying.
    min_close_position : float
        For bullish: close must be in the top X% of the range.
        For bearish: close must be in the bottom X% of the range.
        Default 0.60 = close must be in top/bottom 40%.

    Returns
    -------
    bool
    """
    if avg_body <= 0 or avg_range <= 0:
        return False
    if direction not in ("bullish", "bearish"):
        raise ValueError(f"direction must be 'bullish' or 'bearish', got '{direction}'")

    body   = _body_size(candle)
    rng    = _candle_range(candle)

    if rng <= 0:
        return False

    # 1. Direction check — candle must close in the right direction
    if direction == "bullish" and not _is_bullish(candle):
        return False
    if direction == "bearish" and not _is_bearish(candle):
        return False

    # 2. Body size — must be significantly larger than average
    if body < body_atr_ratio * avg_body:
        return False

    # 3. Body ratio — must be mostly body (not mostly wicks)
    if (body / rng) < min_body_ratio:
        return False

    # 4. Close position — must close with conviction
    #    close_position: 0 = closed at low, 1 = closed at high
    close_position = (candle["close"] - candle["low"]) / rng
    if direction == "bullish" and close_position < min_close_position:
        return False
    if direction == "bearish" and close_position > (1.0 - min_close_position):
        return False

    return True


def displacement_strength(
    candle: pd.Series,
    avg_body: float,
    avg_range: float,
) -> float:
    """
    Returns a score from 0.0 to 1.0 representing how strong the displacement is.

    Combines:
      - Body size multiple (how much bigger than average)
      - Body-to-range ratio (how clean the move is)
      - Close position conviction (how close to extreme of range)

    Used for ranking multiple displacements or setting confidence thresholds.
    """
    if avg_body <= 0 or avg_range <= 0:
        return 0.0

    body = _body_size(candle)
    rng  = _candle_range(candle)

    if rng <= 0:
        return 0.0

    # Component 1: body multiple (capped at 4× to keep score bounded)
    body_multiple = min(body / avg_body, 4.0) / 4.0  # 0–1

    # Component 2: body-to-range ratio (already 0–1)
    body_ratio = body / rng

    # Component 3: close conviction
    close_pos = (candle["close"] - candle["low"]) / rng
    if _is_bullish(candle):
        conviction = close_pos          # 1.0 = closed at high
    else:
        conviction = 1.0 - close_pos   # 1.0 = closed at low

    # Weighted average
    score = (body_multiple * 0.4) + (body_ratio * 0.35) + (conviction * 0.25)
    return round(min(max(score, 0.0), 1.0), 3)


# ── bulk scan ──────────────────────────────────────────────────────────────────

def find_displacements(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 50,
    body_atr_ratio: float = 1.5,
    min_body_ratio: float = 0.55,
    min_close_position: float = 0.60,
    avg_period: int = 20,
) -> list:
    """
    Scan the last `lookback` candles for displacement candles.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    direction : str
        "bullish" or "bearish".
    lookback : int
        How many recent candles to scan. Default 50.
    body_atr_ratio : float
        See is_displacement().
    min_body_ratio : float
        See is_displacement().
    min_close_position : float
        See is_displacement().
    avg_period : int
        Rolling window for average body/range calculation.

    Returns
    -------
    list of dict, most recent first. Each dict:
        {
          "direction":  "bullish" | "bearish",
          "timestamp":  pd.Timestamp,
          "index":      int,           # iloc position
          "open":       float,
          "high":       float,
          "low":        float,
          "close":      float,
          "body":       float,
          "body_ratio": float,         # body / range
          "strength":   float,         # 0.0–1.0
          "avg_body":   float,         # avg body at time of candle
        }
    """
    if df is None or len(df) < avg_period + 3:
        return []
    if direction not in ("bullish", "bearish"):
        raise ValueError(f"direction must be 'bullish' or 'bearish', got '{direction}'")

    avg_bodies = _rolling_avg_body(df, avg_period)
    avg_ranges = _rolling_avg_range(df, avg_period)

    start = max(avg_period, len(df) - lookback)
    displacements = []

    for i in range(start, len(df)):
        candle   = df.iloc[i]
        avg_body = avg_bodies.iloc[i]
        avg_range = avg_ranges.iloc[i]

        if pd.isna(avg_body) or pd.isna(avg_range):
            continue

        if is_displacement(candle, avg_body, avg_range, direction,
                           body_atr_ratio, min_body_ratio, min_close_position):
            rng = _candle_range(candle)
            displacements.append({
                "direction":  direction,
                "timestamp":  df.index[i],
                "index":      i,
                "open":       float(candle["open"]),
                "high":       float(candle["high"]),
                "low":        float(candle["low"]),
                "close":      float(candle["close"]),
                "body":       float(_body_size(candle)),
                "body_ratio": float(_body_size(candle) / rng) if rng > 0 else 0.0,
                "strength":   displacement_strength(candle, avg_body, avg_range),
                "avg_body":   float(avg_body),
            })

    return list(reversed(displacements))


def get_latest_displacement(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 50,
    body_atr_ratio: float = 1.5,
    min_body_ratio: float = 0.55,
    min_close_position: float = 0.60,
    avg_period: int = 20,
    min_strength: float = 0.0,
) -> Optional[dict]:
    """
    Return the single most recent displacement candle, or None.

    Parameters
    ----------
    min_strength : float
        Optional minimum strength score (0.0–1.0).
        Use 0.4+ for higher quality setups only.
    """
    displacements = find_displacements(
        df, direction, lookback,
        body_atr_ratio, min_body_ratio, min_close_position, avg_period
    )

    if not displacements:
        return None

    if min_strength > 0:
        displacements = [d for d in displacements if d["strength"] >= min_strength]

    return displacements[0] if displacements else None


# ── FVG linkage ────────────────────────────────────────────────────────────────

def displacement_preceded_fvg(
    df: pd.DataFrame,
    fvg: dict,
    direction: str,
    lookback_before_fvg: int = 5,
    **kwargs,
) -> bool:
    """
    Returns True if a valid displacement candle exists within
    `lookback_before_fvg` candles BEFORE the FVG impulse candle.

    This confirms the FVG was created by real institutional displacement,
    not random price movement.

    Parameters
    ----------
    fvg : dict
        FVG dict from fvg_detector.get_latest_fvg().
    lookback_before_fvg : int
        How many candles before the FVG impulse to look for displacement.
        Default 5 — the displacement should be very close to the FVG.
    """
    if fvg is None or df is None:
        return False

    impulse_idx = fvg["candle_index"]

    # Include enough history before the search window for the rolling avg to be valid
    avg_period = kwargs.get("avg_period", 20)
    history_start = max(0, impulse_idx - avg_period - lookback_before_fvg)
    search_end    = impulse_idx + 1  # inclusive of impulse candle

    df_slice = df.iloc[history_start:search_end]

    if len(df_slice) < avg_period:
        return False

    # Only look for displacement in the last lookback_before_fvg candles of the slice
    result = get_latest_displacement(
        df_slice, direction,
        lookback=lookback_before_fvg + 1,
        **kwargs
    )
    return result is not None


# ── summary ────────────────────────────────────────────────────────────────────

def summarise_displacements(displacements: list) -> None:
    """Print a human-readable summary. Used for debugging."""
    if not displacements:
        print("  No displacements found.")
        return
    print(f"  Found {len(displacements)} displacement(s):")
    for i, d in enumerate(displacements[:5]):
        print(
            f"    [{i+1}] {d['direction'].upper():8s} | "
            f"{d['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
            f"Body: {d['body']:.2f} (avg: {d['avg_body']:.2f}) | "
            f"Ratio: {d['body_ratio']:.2f} | "
            f"Strength: {d['strength']:.3f}"
        )