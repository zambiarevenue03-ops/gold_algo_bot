import pandas as pd
import numpy as np
from typing import Optional, Tuple
import sys
import os

# ── imports: shared swing bot modules ─────────────────────────────────────────
try:
    from indicators.atr    import calculate_atr, atr_moving_average
    from indicators.obv    import calculate_obv, obv_moving_average
    from indicators.pivots import detect_swings
    from smc.structure     import determine_trend
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from indicators.atr    import calculate_atr, atr_moving_average
    from indicators.obv    import calculate_obv, obv_moving_average
    from indicators.pivots import detect_swings
    from smc.structure     import determine_trend

# ── imports: new scalping modules ─────────────────────────────────────────────
try:
    from scalping.fvg_detector  import get_latest_fvg, price_in_fvg, price_approaching_fvg
    from scalping.displacement  import get_latest_displacement, displacement_preceded_fvg
    from scalping.choch_15m     import choch_aligns_with_bias
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from fvg_detector  import get_latest_fvg, price_in_fvg, price_approaching_fvg
    from displacement  import get_latest_displacement, displacement_preceded_fvg
    from choch_15m     import choch_aligns_with_bias


# ── rejection reason constants ─────────────────────────────────────────────────
class Rejection:
    NONE                   = None
    HTF_TREND_MISMATCH     = "htf_trend_mismatch"       # 4H and 1H trend disagree
    HTF_TREND_UNCLEAR      = "htf_trend_unclear"        # Not enough swings on 4H/1H
    OBV_AGAINST_BIAS       = "obv_against_bias"         # OBV moving opposite to trend
    ATR_BELOW_MA           = "atr_below_ma"             # Market not moving enough
    NO_DISPLACEMENT        = "no_displacement"          # No impulse candle on 15M
    NO_FVG                 = "no_fvg"                   # No Fair Value Gap found
    FVG_MITIGATED          = "fvg_mitigated"            # FVG already fully filled
    PRICE_NOT_IN_FVG       = "price_not_in_fvg"         # Price not in or near FVG
    SOFT_SCORE_TOO_LOW     = "soft_score_too_low"        # Confirmation score < min
    OUTSIDE_SESSION        = "outside_session"          # Not in London or NY session


# ── session filter ─────────────────────────────────────────────────────────────

def _is_trading_session(timestamp: pd.Timestamp, utc_offset_hours: int = 0) -> bool:
    """
    Returns True if timestamp falls within active trading sessions.

    Sessions (UTC):
      London open:  07:00 – 12:00
      NY open:      13:00 – 17:00
      (Gap 12:00–13:00 excluded — lunch lull, lower volume)
      Pre-London:   06:00 – 07:00 (optional — liquidity sweeps happen here)

    Parameters
    ----------
    utc_offset_hours : int
        Offset to apply if your CSV timestamps are in broker local time.
        Default 0 = timestamps are already UTC.
        Example: +2 if broker is UTC+2 (EET) — common for MT5 brokers.
    """
    ts = timestamp
    if utc_offset_hours != 0:
        ts = ts - pd.Timedelta(hours=utc_offset_hours)

    hour = ts.hour

    # London open (including pre-London)
    if 6 <= hour < 12:
        return True
    # NY open
    if 13 <= hour < 17:
        return True

    return False


# ── layer helpers ──────────────────────────────────────────────────────────────

def _get_htf_trend(df: pd.DataFrame, pivot_lookback: int) -> Optional[str]:
    """Get trend from a single timeframe using swing detection."""
    if df is None or len(df) < pivot_lookback * 4:
        return None
    sh, sl = detect_swings(df, lookback=pivot_lookback)
    return determine_trend(sh, sl)


def _obv_aligned(df: pd.DataFrame, bias: str, obv_ma_period: int) -> bool:
    """Returns True if OBV direction agrees with bias."""
    if df is None or len(df) < obv_ma_period + 5:
        return False
    obv    = calculate_obv(df)
    obv_ma = obv_moving_average(obv, obv_ma_period)
    if obv.iloc[-1] is None or obv_ma.iloc[-1] is None:
        return False
    if bias == "bullish":
        return float(obv.iloc[-1]) > float(obv_ma.iloc[-1])
    return float(obv.iloc[-1]) < float(obv_ma.iloc[-1])


def _atr_expanding(df: pd.DataFrame, atr_period: int, atr_ma_period: int) -> bool:
    """Returns True if ATR is above its moving average (market is expanding)."""
    if df is None or len(df) < atr_period + atr_ma_period:
        return False
    atr    = calculate_atr(df, atr_period)
    atr_ma = atr_moving_average(atr, atr_ma_period)
    if pd.isna(atr.iloc[-1]) or pd.isna(atr_ma.iloc[-1]):
        return False
    return float(atr.iloc[-1]) > float(atr_ma.iloc[-1])


# ── main entry function ────────────────────────────────────────────────────────

def check_entry_signal(
    df_1h:  pd.DataFrame,
    df_4h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    # HTF parameters
    pivot_lookback:       int   = 2,
    obv_ma_period:        int   = 10,
    atr_period:           int   = 14,
    atr_ma_period:        int   = 14,
    # 15M setup parameters
    fvg_lookback:         int   = 50,
    fvg_min_gap_ratio:    float = 0.1,
    disp_body_ratio:      float = 1.5,
    disp_min_body_pct:    float = 0.55,
    disp_lookback:        int   = 30,
    # CHoCH parameters
    choch_max_candles_ago: int  = 15,
    # Soft score threshold
    min_soft_score:       int   = 1,
    # Session filter
    session_filter:       bool  = True,
    utc_offset_hours:     int   = 0,
) -> Optional[str]:
    """
    Check for a valid scalp entry signal.

    Parameters
    ----------
    df_1h  : pd.DataFrame  — 1H OHLCV data (used for HTF trend)
    df_4h  : pd.DataFrame  — 4H OHLCV data (used for HTF trend + OBV + ATR)
    df_15m : pd.DataFrame  — 15M OHLCV data (used for FVG + displacement + CHoCH)

    Returns
    -------
    "long"  — bullish entry signal
    "short" — bearish entry signal
    None    — no valid signal
    """
    signal, _, _ = check_entry_signal_detailed(
        df_1h, df_4h, df_15m,
        pivot_lookback, obv_ma_period, atr_period, atr_ma_period,
        fvg_lookback, fvg_min_gap_ratio,
        disp_body_ratio, disp_min_body_pct, disp_lookback,
        choch_max_candles_ago, min_soft_score,
        session_filter, utc_offset_hours,
    )
    return signal


def check_entry_signal_detailed(
    df_1h:  pd.DataFrame,
    df_4h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    pivot_lookback:        int   = 2,
    obv_ma_period:         int   = 10,
    atr_period:            int   = 14,
    atr_ma_period:         int   = 14,
    fvg_lookback:          int   = 50,
    fvg_min_gap_ratio:     float = 0.1,
    disp_body_ratio:       float = 1.5,
    disp_min_body_pct:     float = 0.55,
    disp_lookback:         int   = 30,
    choch_max_candles_ago: int   = 15,
    min_soft_score:        int   = 1,
    session_filter:        bool  = True,
    utc_offset_hours:      int   = 0,
) -> Tuple[Optional[str], Optional[str], dict]:
    """
    Full entry check with detailed rejection reason and layer scores.

    Returns
    -------
    (signal, rejection_reason, layer_scores)

    signal          : "long" | "short" | None
    rejection_reason: Rejection.* constant string, or None if signal found
    layer_scores    : dict with all intermediate results for debugging
    """
    scores = {
        "trend_4h":       None,
        "trend_1h":       None,
        "bias":           None,
        "obv_aligned":    False,
        "atr_expanding":  False,
        "displacement":   None,
        "fvg":            None,
        "price_in_fvg":   False,
        "choch":          False,
        "soft_score":     0,
        "session_ok":     False,
    }

    # ── LAYER 1A: HTF TREND ───────────────────────────────────────────────────
    trend_4h = _get_htf_trend(df_4h, pivot_lookback)
    trend_1h = _get_htf_trend(df_1h, pivot_lookback)

    scores["trend_4h"] = trend_4h
    scores["trend_1h"] = trend_1h

    if trend_4h is None or trend_1h is None:
        return None, Rejection.HTF_TREND_UNCLEAR, scores

    if trend_4h != trend_1h:
        return None, Rejection.HTF_TREND_MISMATCH, scores

    bias = trend_4h  # "bullish" or "bearish"
    scores["bias"] = bias

    # ── LAYER 1B: OBV CONFIRMATION ────────────────────────────────────────────
    obv_ok = _obv_aligned(df_4h, bias, obv_ma_period)
    scores["obv_aligned"] = obv_ok

    if not obv_ok:
        return None, Rejection.OBV_AGAINST_BIAS, scores

    # ── LAYER 1C: ATR EXPANSION ───────────────────────────────────────────────
    atr_ok = _atr_expanding(df_4h, atr_period, atr_ma_period)
    scores["atr_expanding"] = atr_ok

    if not atr_ok:
        return None, Rejection.ATR_BELOW_MA, scores

    # ── LAYER 2A: DISPLACEMENT ON 15M ────────────────────────────────────────
    disp = get_latest_displacement(
        df_15m,
        direction=bias,
        lookback=disp_lookback,
        body_atr_ratio=disp_body_ratio,
        min_body_ratio=disp_min_body_pct,
    )
    scores["displacement"] = disp

    if disp is None:
        return None, Rejection.NO_DISPLACEMENT, scores

    # ── LAYER 2B: FVG ON 15M ─────────────────────────────────────────────────
    fvg = get_latest_fvg(
        df_15m,
        direction=bias,
        lookback=fvg_lookback,
        min_gap_atr_ratio=fvg_min_gap_ratio,
        exclude_mitigated=True,
    )
    scores["fvg"] = fvg

    if fvg is None:
        return None, Rejection.NO_FVG, scores

    # ── LAYER 2C: PRICE IN OR APPROACHING FVG ────────────────────────────────
    current_price = float(df_15m["close"].iloc[-1])
    in_fvg        = price_in_fvg(current_price, fvg)
    approaching   = price_approaching_fvg(current_price, fvg)

    scores["price_in_fvg"] = bool(in_fvg or approaching)

    if not (in_fvg or approaching):
        return None, Rejection.PRICE_NOT_IN_FVG, scores

    # ── LAYER 3: SOFT CONFIRMATIONS ───────────────────────────────────────────
    soft_score = 0

    # Soft 1: CHoCH aligns with bias on 15M
    choch_ok = choch_aligns_with_bias(
        df_15m,
        htf_bias=bias,
        max_candles_ago=choch_max_candles_ago,
        pivot_lookback=pivot_lookback,
    )
    if choch_ok:
        soft_score += 1
    scores["choch"] = choch_ok

    # Soft 2: Displacement is recent (within last 30 candles)
    disp_candles_ago = (len(df_15m) - 1) - disp["index"]
    if disp_candles_ago <= 20:
        soft_score += 1

    # Soft 3: Displacement strength is high
    if disp.get("strength", 0) >= 0.5:
        soft_score += 1

    scores["soft_score"] = soft_score

    if soft_score < min_soft_score:
        return None, Rejection.SOFT_SCORE_TOO_LOW, scores

    # ── LAYER 4: SESSION FILTER ───────────────────────────────────────────────
    if session_filter:
        current_time = df_15m.index[-1]
        session_ok   = _is_trading_session(current_time, utc_offset_hours)
        scores["session_ok"] = session_ok

        if not session_ok:
            return None, Rejection.OUTSIDE_SESSION, scores
    else:
        scores["session_ok"] = True

    # ── ALL FILTERS PASSED ────────────────────────────────────────────────────
    signal = "long" if bias == "bullish" else "short"
    return signal, Rejection.NONE, scores


# ── rejection frequency tracker ───────────────────────────────────────────────

class RejectionTracker:
    """
    Tracks how often each rejection reason fires.
    Use this during backtesting to diagnose what is blocking signals.

    Usage:
        tracker = RejectionTracker()
        signal, reason, _ = check_entry_signal_detailed(...)
        tracker.record(reason)

        tracker.summary()   # prints table sorted by frequency
    """

    def __init__(self):
        self._counts: dict = {}
        self._total:  int  = 0

    def record(self, reason: Optional[str]) -> None:
        key = reason if reason is not None else "SIGNAL_GENERATED"
        self._counts[key] = self._counts.get(key, 0) + 1
        self._total += 1

    def summary(self) -> None:
        if self._total == 0:
            print("  No data recorded yet.")
            return
        print(f"\n  {'Rejection Reason':<30} {'Count':>8} {'%':>8}")
        print(f"  {'-'*50}")
        for reason, count in sorted(self._counts.items(), key=lambda x: -x[1]):
            pct = count / self._total * 100
            marker = " * SIGNAL" if reason == "SIGNAL_GENERATED" else ""
            print(f"  {reason:<30} {count:>8,} {pct:>7.1f}%{marker}")
        print(f"  {'-'*50}")
        print(f"  {'TOTAL':<30} {self._total:>8,}")

    def reset(self) -> None:
        self._counts = {}
        self._total  = 0

    def get_signal_rate(self) -> float:
        if self._total == 0:
            return 0.0
        signals = self._counts.get("SIGNAL_GENERATED", 0)
        return signals / self._total
