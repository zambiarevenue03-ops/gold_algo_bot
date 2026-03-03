"""
scalping/risk_model.py
=======================
Risk management for the XAUUSD scalping bot.

Responsibilities:
  1. Stop Loss placement   — structural, based on FVG zone + buffer
  2. Take Profit levels    — 3-tiered (TP1 / TP2 / TP3)
  3. Position sizing       — fixed % risk per trade
  4. Trade parameters      — single dict ready for execution
  5. Risk validation       — sanity checks before any order is placed

Stop Loss Logic:
  Long  → SL below the FVG low (or displacement low, whichever is lower)
          plus a buffer of sl_buffer_atr × ATR to avoid false triggers
  Short → SL above the FVG high (or displacement high, whichever is higher)
          plus the same buffer

Take Profit Logic (3-tiered partial exit):
  TP1 (50% of position) → entry ± tp1_r × risk  (default 1.5R)
  TP2 (30% of position) → entry ± tp2_r × risk  (default 2.5R)
  TP3 (20% — runner)    → entry ± tp3_r × risk  (default 4.0R)
                          TP3 uses trailing stop after TP1 hit

Position Sizing:
  lot_size = (account_balance × risk_pct) / (sl_distance × point_value)
  Capped by max_lot_size to prevent over-sizing on tight SLs.

Public API:
    calculate_stop_loss(signal, entry, fvg, displacement, atr, params)
        → float

    calculate_take_profits(signal, entry, stop_loss, params)
        → dict {tp1, tp2, tp3}

    calculate_position_size(account_balance, risk_pct, entry, stop_loss, params)
        → float (lot size)

    build_trade_parameters(signal, entry, fvg, displacement, atr,
                           account_balance, params)
        → dict (complete trade spec) | None (if invalid)

    validate_trade(trade_params, params)
        → (bool, str)   (valid, reason)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field


# -- default parameters ---------------------------------------------------------

@dataclass
class ScalpRiskParams:
    """
    All risk parameters for the scalping bot in one place.
    Override individual fields as needed.
    """
    # Position sizing
    risk_pct:          float = 0.0025   # 0.25% of account per trade
    max_risk_pct:      float = 0.005    # Hard cap: never more than 0.5%
    min_lot_size:      float = 0.01     # Minimum tradeable lot (MT5)
    max_lot_size:      float = 5.0      # Maximum allowed lot size
    point_value:       float = 10.0     # USD per pip per lot (XAUUSD standard)
    pip_size:          float = 0.1      # 1 pip = $0.10 on XAUUSD

    # Stop loss
    sl_buffer_atr:     float = 0.3      # Buffer below/above structure (× ATR)
    sl_min_distance:   float = 3.0      # Minimum SL distance in price points ($3)
    sl_max_distance:   float = 50.0     # Maximum SL distance ($50) — if wider, skip

    # Take profit R-multiples
    tp1_r:             float = 2.0      
    tp2_r:             float = 3.0      
    tp3_r:             float = 3.0      

    # Partial exit sizes (must sum to 1.0)
    tp1_pct:           float = 0.50     
    tp2_pct:           float = 0.50     
    tp3_pct:           float = 0.00     

    # Minimum R:R ratio (safety check)
    min_rr:            float = 1.5      # Don't take trades with R:R below 1.5


DEFAULT_PARAMS = ScalpRiskParams()


# -- stop loss ------------------------------------------------------------------

def calculate_stop_loss(
    signal:      str,
    entry:       float,
    fvg:         Optional[dict],
    displacement: Optional[dict],
    atr:         float,
    params:      ScalpRiskParams = DEFAULT_PARAMS,
) -> Optional[float]:
    """
    Calculate the structural stop loss level.

    Logic:
      Long:  SL = min(fvg_low, displacement_low) - buffer
      Short: SL = max(fvg_high, displacement_high) + buffer

    The SL is anchored to the STRUCTURE that created the setup,
    not to a fixed pip distance. This is intentional — if the structure
    is invalidated (price returns below the FVG and displacement), the
    trade thesis is wrong.

    Parameters
    ----------
    signal       : "long" or "short"
    entry        : current price (close of entry candle)
    fvg          : FVG dict from fvg_detector (may be None)
    displacement : displacement dict from displacement.py (may be None)
    atr          : ATR value on 15M at time of entry
    params       : ScalpRiskParams

    Returns
    -------
    float — stop loss price, or None if calculation is not possible
    """
    if signal not in ("long", "short"):
        return None

    buffer = params.sl_buffer_atr * atr

    if signal == "long":
        # Gather candidate levels (the lowest point of the setup structure)
        candidates = []

        if fvg is not None:
            candidates.append(fvg["fvg_low"])

        if displacement is not None:
            candidates.append(displacement["low"])

        if not candidates:
            # Fallback: use minimum SL distance from entry
            return round(entry - params.sl_min_distance - buffer, 2)

        # SL goes below the lowest structural level
        structure_low = min(candidates)
        sl = structure_low - buffer
        return round(sl, 2)

    else:  # short
        candidates = []

        if fvg is not None:
            candidates.append(fvg["fvg_high"])

        if displacement is not None:
            candidates.append(displacement["high"])

        if not candidates:
            return round(entry + params.sl_min_distance + buffer, 2)

        structure_high = max(candidates)
        sl = structure_high + buffer
        return round(sl, 2)


# -- take profits ---------------------------------------------------------------

def calculate_take_profits(
    signal:    str,
    entry:     float,
    stop_loss: float,
    params:    ScalpRiskParams = DEFAULT_PARAMS,
) -> Optional[dict]:
    """
    Calculate all three take profit levels based on R-multiples.

    Parameters
    ----------
    signal    : "long" or "short"
    entry     : entry price
    stop_loss : stop loss price
    params    : ScalpRiskParams

    Returns
    -------
    dict:
        {
          "risk":     float,  # distance from entry to SL (1R in price)
          "tp1":      float,  # 1.5R target (50% exit)
          "tp2":      float,  # 2.5R target (30% exit)
          "tp3":      float,  # 4.0R target (20% runner)
          "tp1_pct":  float,  # fraction of position to close at TP1
          "tp2_pct":  float,
          "tp3_pct":  float,
          "tp1_r":    float,  # actual R multiple
          "tp2_r":    float,
          "tp3_r":    float,
        }
    None if inputs are invalid.
    """
    if signal not in ("long", "short"):
        return None
    if stop_loss is None:
        return None

    risk = abs(entry - stop_loss)
    if risk <= 0:
        return None

    if signal == "long":
        tp1 = round(entry + params.tp1_r * risk, 2)
        tp2 = round(entry + params.tp2_r * risk, 2)
        tp3 = round(entry + params.tp3_r * risk, 2)
    else:  # short
        tp1 = round(entry - params.tp1_r * risk, 2)
        tp2 = round(entry - params.tp2_r * risk, 2)
        tp3 = round(entry - params.tp3_r * risk, 2)

    return {
        "risk":    round(risk, 2),
        "tp1":     tp1,
        "tp2":     tp2,
        "tp3":     tp3,
        "tp1_pct": params.tp1_pct,
        "tp2_pct": params.tp2_pct,
        "tp3_pct": params.tp3_pct,
        "tp1_r":   params.tp1_r,
        "tp2_r":   params.tp2_r,
        "tp3_r":   params.tp3_r,
    }


# -- position sizing ------------------------------------------------------------

def calculate_position_size(
    account_balance: float,
    entry:           float,
    stop_loss:       float,
    params:          ScalpRiskParams = DEFAULT_PARAMS,
    risk_pct:        Optional[float] = None,
) -> Optional[float]:
    """
    Calculate lot size using fixed fractional risk model.

    Formula:
        risk_amount   = account_balance × risk_pct
        sl_pips       = |entry - stop_loss| / pip_size
        lot_size      = risk_amount / (sl_pips × point_value)

    For XAUUSD:
        pip_size    = 0.1  (1 pip = $0.10 move)
        point_value = $10  (per pip per standard lot)
        So: 1 pip of risk on 1 lot = $10

    Parameters
    ----------
    account_balance : float — account equity in USD
    entry           : float — entry price
    stop_loss       : float — stop loss price
    params          : ScalpRiskParams
    risk_pct        : float (optional) — override params.risk_pct

    Returns
    -------
    float — lot size rounded to 2 decimal places
    None  — if inputs are invalid
    """
    if account_balance <= 0 or entry <= 0 or stop_loss is None:
        return None

    sl_distance = abs(entry - stop_loss)
    if sl_distance <= 0:
        return None

    pct = risk_pct if risk_pct is not None else params.risk_pct
    pct = min(pct, params.max_risk_pct)   # enforce hard cap

    risk_amount = account_balance * pct
    sl_pips     = sl_distance / params.pip_size
    lot_size    = risk_amount / (sl_pips * params.point_value)

    # Clamp to allowed range
    lot_size = max(params.min_lot_size, min(lot_size, params.max_lot_size))

    return round(lot_size, 2)


# -- full trade parameters ------------------------------------------------------

def build_trade_parameters(
    signal:          str,
    entry:           float,
    fvg:             Optional[dict],
    displacement:    Optional[dict],
    atr:             float,
    account_balance: float,
    params:          ScalpRiskParams = DEFAULT_PARAMS,
    timestamp:       Optional[pd.Timestamp] = None,
) -> Optional[dict]:
    """
    Build the complete trade specification from a confirmed entry signal.

    This is the single function the backtest engine and live trader call
    after check_entry_signal() returns a signal.

    Parameters
    ----------
    signal          : "long" or "short"
    entry           : entry price (close of entry candle)
    fvg             : FVG dict (from fvg_detector)
    displacement    : displacement dict (from displacement.py)
    atr             : ATR on 15M at entry time
    account_balance : current account equity
    params          : ScalpRiskParams
    timestamp       : entry timestamp (for logging)

    Returns
    -------
    dict — complete trade spec, or None if trade is invalid
    {
      "signal":          "long" | "short",
      "entry":           float,
      "stop_loss":       float,
      "tp1":             float,
      "tp2":             float,
      "tp3":             float,
      "risk":            float,   # 1R in price
      "risk_reward_tp1": float,   # R:R to TP1
      "risk_reward_tp2": float,   # R:R to TP2
      "lot_size":        float,
      "risk_amount_usd": float,   # $ at risk
      "tp1_pct":         float,
      "tp2_pct":         float,
      "tp3_pct":         float,
      "fvg_low":         float | None,
      "fvg_high":        float | None,
      "timestamp":       pd.Timestamp | None,
    }
    """
    if signal not in ("long", "short"):
        return None

    # 1. Stop loss
    sl = calculate_stop_loss(signal, entry, fvg, displacement, atr, params)
    if sl is None:
        return None

    # 2. Take profits
    tps = calculate_take_profits(signal, entry, sl, params)
    if tps is None:
        return None

    # 3. Position size
    lot = calculate_position_size(account_balance, entry, sl, params)
    if lot is None:
        return None

    # 4. Risk amount in USD
    sl_distance  = abs(entry - sl)
    sl_pips      = sl_distance / params.pip_size
    risk_amount  = sl_pips * params.point_value * lot

    # 5. R:R ratios
    rr_tp1 = tps["tp1_r"]
    rr_tp2 = tps["tp2_r"]

    # 6. Validate
    valid, reason = validate_trade(
        signal=signal,
        entry=entry,
        stop_loss=sl,
        lot_size=lot,
        risk=tps["risk"],
        params=params,
    )
    if not valid:
        return None

    return {
        "signal":           signal,
        "entry":            round(entry, 2),
        "stop_loss":        sl,
        "tp1":              tps["tp1"],
        "tp2":              tps["tp2"],
        "tp3":              tps["tp3"],
        "risk":             tps["risk"],
        "risk_reward_tp1":  rr_tp1,
        "risk_reward_tp2":  rr_tp2,
        "lot_size":         lot,
        "risk_amount_usd":  round(risk_amount, 2),
        "tp1_pct":          tps["tp1_pct"],
        "tp2_pct":          tps["tp2_pct"],
        "tp3_pct":          tps["tp3_pct"],
        "fvg_low":          fvg["fvg_low"]  if fvg else None,
        "fvg_high":         fvg["fvg_high"] if fvg else None,
        "timestamp":        timestamp,
    }


# -- validation -----------------------------------------------------------------

def validate_trade(
    signal:    str,
    entry:     float,
    stop_loss: float,
    lot_size:  float,
    risk:      float,
    params:    ScalpRiskParams = DEFAULT_PARAMS,
) -> Tuple[bool, str]:
    """
    Final sanity check before placing any order.

    Returns (True, "") if valid.
    Returns (False, reason) if any check fails.
    """
    # SL on wrong side of entry
    if signal == "long" and stop_loss >= entry:
        return False, f"Long SL ({stop_loss}) is above or at entry ({entry})"
    if signal == "short" and stop_loss <= entry:
        return False, f"Short SL ({stop_loss}) is below or at entry ({entry})"

    # SL too tight
    if risk < params.sl_min_distance:
        return False, f"SL distance {risk:.2f} below minimum {params.sl_min_distance}"

    # SL too wide
    if risk > params.sl_max_distance:
        return False, f"SL distance {risk:.2f} exceeds maximum {params.sl_max_distance}"

    # Lot size out of range
    if lot_size < params.min_lot_size:
        return False, f"Lot {lot_size} below minimum {params.min_lot_size}"
    if lot_size > params.max_lot_size:
        return False, f"Lot {lot_size} exceeds maximum {params.max_lot_size}"

    # R:R check (TP1 is minimum acceptable)
    if params.tp1_r < params.min_rr:
        return False, f"TP1 R:R {params.tp1_r} below minimum {params.min_rr}"

    return True, ""


# -- daily risk guard -----------------------------------------------------------

class DailyRiskGuard:
    """
    Tracks intraday P&L and enforces daily/weekly loss limits.
    Must be called after every trade closes.

    Kill switch rules:
      - Daily loss >= 2% of starting balance  → stop for the day
      - Weekly loss >= 5% of starting balance → stop for the week
      - 3 consecutive losses                  → pause 2 hours
    """

    def __init__(
        self,
        starting_balance:    float,
        max_daily_loss_pct:  float = 0.02,
        max_weekly_loss_pct: float = 0.05,
        max_consecutive_losses: int = 3,
    ):
        self.starting_balance       = starting_balance
        self.daily_start_balance    = starting_balance
        self.weekly_start_balance   = starting_balance
        self.max_daily_loss_pct     = max_daily_loss_pct
        self.max_weekly_loss_pct    = max_weekly_loss_pct
        self.max_consecutive_losses = max_consecutive_losses

        self.daily_pnl              = 0.0
        self.weekly_pnl             = 0.0
        self.consecutive_losses     = 0
        self.paused_until:          Optional[pd.Timestamp] = None
        self.daily_stopped:         bool = False
        self.weekly_stopped:        bool = False

    def record_trade(self, pnl_usd: float, timestamp: pd.Timestamp) -> None:
        """Call this after every trade closes with the P&L in USD."""
        self.daily_pnl  += pnl_usd
        self.weekly_pnl += pnl_usd

        if pnl_usd < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Pause after N consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.paused_until = timestamp + pd.Timedelta(hours=2)
            self.consecutive_losses = 0  # reset after pause

    def can_trade(self, current_balance: float, timestamp: pd.Timestamp) -> Tuple[bool, str]:
        """
        Returns (True, "") if trading is allowed.
        Returns (False, reason) if kill switch is active.
        """
        # Check pause
        if self.paused_until and timestamp < self.paused_until:
            return False, f"Paused until {self.paused_until} after consecutive losses"

        # Daily loss limit
        daily_loss_pct = self.daily_pnl / self.daily_start_balance
        if daily_loss_pct <= -self.max_daily_loss_pct:
            self.daily_stopped = True
            return False, f"Daily loss limit hit ({daily_loss_pct*100:.2f}%)"

        # Weekly loss limit
        weekly_loss_pct = self.weekly_pnl / self.weekly_start_balance
        if weekly_loss_pct <= -self.max_weekly_loss_pct:
            self.weekly_stopped = True
            return False, f"Weekly loss limit hit ({weekly_loss_pct*100:.2f}%)"

        return True, ""

    def new_day(self, current_balance: float) -> None:
        """Call at the start of each new trading day."""
        self.daily_start_balance = current_balance
        self.daily_pnl           = 0.0
        self.daily_stopped       = False
        self.consecutive_losses  = 0

    def new_week(self, current_balance: float) -> None:
        """Call at the start of each new trading week."""
        self.new_day(current_balance)
        self.weekly_start_balance = current_balance
        self.weekly_pnl           = 0.0
        self.weekly_stopped       = False

    def summary(self) -> dict:
        return {
            "daily_pnl":          round(self.daily_pnl, 2),
            "weekly_pnl":         round(self.weekly_pnl, 2),
            "consecutive_losses": self.consecutive_losses,
            "daily_stopped":      self.daily_stopped,
            "weekly_stopped":     self.weekly_stopped,
            "paused_until":       str(self.paused_until) if self.paused_until else None,
        }


# -- display helpers ------------------------------------------------------------

def summarise_trade(trade: Optional[dict]) -> None:
    """Print a human-readable trade summary. Used for debugging."""
    if trade is None:
        print("  Trade: INVALID (None)")
        return
    direction = trade["signal"].upper()
    print(f"\n  {'-'*45}")
    print(f"  {direction} TRADE PARAMETERS")
    print(f"  {'-'*45}")
    print(f"  Entry:       {trade['entry']:.2f}")
    print(f"  Stop Loss:   {trade['stop_loss']:.2f}  (risk: ${trade['risk']:.2f})")
    print(f"  TP1 (1.5R):  {trade['tp1']:.2f}  (50% exit)")
    print(f"  TP2 (2.5R):  {trade['tp2']:.2f}  (30% exit)")
    print(f"  TP3 (4.0R):  {trade['tp3']:.2f}  (20% runner)")
    print(f"  Lot Size:    {trade['lot_size']:.2f}")
    print(f"  Risk $:      ${trade['risk_amount_usd']:.2f}")
    print(f"  R:R to TP2:  1:{trade['risk_reward_tp2']:.1f}")
    print(f"  {'-'*45}\n")