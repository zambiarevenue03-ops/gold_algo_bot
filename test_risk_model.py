"""
test_risk_model.py
===================
Tests for scalping/risk_model.py

Covers:
  - Stop loss placement (long + short + edge cases)
  - Take profit levels (3-tier + R-multiple verification)
  - Position sizing (formula accuracy + caps)
  - build_trade_parameters (end-to-end)
  - validate_trade (all failure modes)
  - DailyRiskGuard (kill switches)

Run from gold_algo_bot/ root:
    python test_risk_model.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from scalping.risk_model import (
    ScalpRiskParams,
    calculate_stop_loss,
    calculate_take_profits,
    calculate_position_size,
    build_trade_parameters,
    validate_trade,
    DailyRiskGuard,
    summarise_trade,
    DEFAULT_PARAMS,
)

# == helpers ====================================================================
GREEN = "\033[92m"; RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"
passed = 0; failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  {GREEN}[PASS]{RESET}  {name}"); passed += 1
    else:
        print(f"  {RED}[FAIL]{RESET}  {RED}{name}{RESET}" + (f"  <- {detail}" if detail else "")); failed += 1

def approx(a, b, tol=0.01):
    return abs(a - b) <= tol

# == fixtures ==================================================================─
ENTRY_LONG  = 2520.00
ENTRY_SHORT = 2520.00
ATR         = 8.0       # $8 ATR on 15M — typical for XAUUSD

FVG_BULL = {
    "type": "bullish", "fvg_low": 2510.0, "fvg_high": 2525.0,
    "gap_size": 15.0, "midpoint": 2517.5,
    "candle_index": 10, "timestamp": pd.Timestamp("2025-01-06 09:00"),
}
FVG_BEAR = {
    "type": "bearish", "fvg_low": 2515.0, "fvg_high": 2530.0,
    "gap_size": 15.0, "midpoint": 2522.5,
    "candle_index": 10, "timestamp": pd.Timestamp("2025-01-06 09:00"),
}
DISP_BULL = {
    "direction": "bullish", "index": 9, "open": 2490.0,
    "high": 2526.0, "low": 2488.0, "close": 2522.0,
    "body": 32.0, "body_ratio": 0.89, "strength": 0.92, "avg_body": 5.0,
    "timestamp": pd.Timestamp("2025-01-06 08:45"),
}
DISP_BEAR = {
    "direction": "bearish", "index": 9, "open": 2550.0,
    "high": 2552.0, "low": 2514.0, "close": 2518.0,
    "body": 32.0, "body_ratio": 0.89, "strength": 0.91, "avg_body": 5.0,
    "timestamp": pd.Timestamp("2025-01-06 08:45"),
}

PARAMS = ScalpRiskParams()
BALANCE = 10_000.0


# ================================================================================
# TEST 1: calculate_stop_loss — Long
# ================================================================================
print(f"\n{BOLD}== TEST 1: Stop Loss — Long =={RESET}")

# SL should be below min(fvg_low, disp_low) minus buffer
# fvg_low=2510, disp_low=2488 -> min=2488, buffer=0.3×8=2.4 -> SL=2485.6
sl_long = calculate_stop_loss("long", ENTRY_LONG, FVG_BULL, DISP_BULL, ATR, PARAMS)
check("Long SL is a float",        sl_long is not None and isinstance(sl_long, float))
check("Long SL is below entry",    sl_long < ENTRY_LONG, f"SL={sl_long}, entry={ENTRY_LONG}")
check("Long SL is below FVG low",  sl_long < FVG_BULL["fvg_low"], f"SL={sl_long}")
check("Long SL is below disp low", sl_long < DISP_BULL["low"], f"SL={sl_long}")
expected_sl_long = DISP_BULL["low"] - (PARAMS.sl_buffer_atr * ATR)
check("Long SL matches formula",   approx(sl_long, expected_sl_long, 0.05),
      f"got {sl_long}, expected {expected_sl_long:.2f}")

# FVG only (no displacement)
sl_fvg_only = calculate_stop_loss("long", ENTRY_LONG, FVG_BULL, None, ATR, PARAMS)
check("Long SL (FVG only) below FVG low", sl_fvg_only < FVG_BULL["fvg_low"])

# No structure -> fallback SL
sl_fallback = calculate_stop_loss("long", ENTRY_LONG, None, None, ATR, PARAMS)
check("Fallback long SL below entry", sl_fallback < ENTRY_LONG)


# ================================================================================
# TEST 2: calculate_stop_loss — Short
# ================================================================================
print(f"\n{BOLD}== TEST 2: Stop Loss — Short =={RESET}")

# fvg_high=2530, disp_high=2552 -> max=2552, buffer=2.4 -> SL=2554.4
sl_short = calculate_stop_loss("short", ENTRY_SHORT, FVG_BEAR, DISP_BEAR, ATR, PARAMS)
check("Short SL is a float",         sl_short is not None)
check("Short SL is above entry",     sl_short > ENTRY_SHORT, f"SL={sl_short}")
check("Short SL is above FVG high",  sl_short > FVG_BEAR["fvg_high"], f"SL={sl_short}")
check("Short SL is above disp high", sl_short > DISP_BEAR["high"], f"SL={sl_short}")
expected_sl_short = DISP_BEAR["high"] + (PARAMS.sl_buffer_atr * ATR)
check("Short SL matches formula",    approx(sl_short, expected_sl_short, 0.05),
      f"got {sl_short}, expected {expected_sl_short:.2f}")

# Edge cases
check("Invalid signal returns None", calculate_stop_loss("sideways", ENTRY_LONG, FVG_BULL, None, ATR) is None)


# ================================================================================
# TEST 3: calculate_take_profits
# ================================================================================
print(f"\n{BOLD}== TEST 3: Take Profits =={RESET}")

sl = 2505.0   # risk = 2520 - 2505 = 15
tps = calculate_take_profits("long", ENTRY_LONG, sl, PARAMS)

check("TPs returned as dict", isinstance(tps, dict))
check("Risk = 15.0",          approx(tps["risk"], 15.0))

# TP levels
check("TP1 = entry + 1.5R",  approx(tps["tp1"], ENTRY_LONG + 1.5 * 15))   # 2542.5
check("TP2 = entry + 2.5R",  approx(tps["tp2"], ENTRY_LONG + 2.5 * 15))   # 2557.5
check("TP3 = entry + 4.0R",  approx(tps["tp3"], ENTRY_LONG + 4.0 * 15))   # 2580.0

# TP order: TP1 < TP2 < TP3 for long
check("Long TP order: TP1 < TP2 < TP3", tps["tp1"] < tps["tp2"] < tps["tp3"])

# Short TPs
sl_s = 2535.0   # risk = 2535 - 2520 = 15
tps_s = calculate_take_profits("short", ENTRY_SHORT, sl_s, PARAMS)
check("Short TP1 = entry - 1.5R", approx(tps_s["tp1"], ENTRY_SHORT - 1.5 * 15))  # 2497.5
check("Short TP order: TP1 > TP2 > TP3", tps_s["tp1"] > tps_s["tp2"] > tps_s["tp3"])

# Partial size sum = 1.0
total_pct = PARAMS.tp1_pct + PARAMS.tp2_pct + PARAMS.tp3_pct
check("TP percentages sum to 1.0", approx(total_pct, 1.0, 0.001),
      f"sum={total_pct}")

# Edge cases
check("Zero risk returns None",   calculate_take_profits("long", 2520, 2520, PARAMS) is None)
check("Invalid signal returns None", calculate_take_profits("up", 2520, 2510, PARAMS) is None)
check("None SL returns None",     calculate_take_profits("long", 2520, None, PARAMS) is None)


# ================================================================================
# TEST 4: calculate_position_size
# ================================================================================
print(f"\n{BOLD}== TEST 4: Position Sizing =={RESET}")

# Manual calculation:
# risk_amount = 10000 × 0.0025 = $25
# sl_distance = |2520 - 2505| = 15
# sl_pips = 15 / 0.1 = 150 pips
# lot_size = 25 / (150 × 10) = 25 / 1500 = 0.0167 -> rounds to 0.02
sl_ps = 2505.0
lot = calculate_position_size(BALANCE, ENTRY_LONG, sl_ps, PARAMS)
check("Lot size is a float",  lot is not None and isinstance(lot, float))
check("Lot size > 0",         lot > 0)
expected_lot = (BALANCE * PARAMS.risk_pct) / ((abs(ENTRY_LONG - sl_ps) / PARAMS.pip_size) * PARAMS.point_value)
check("Lot size matches formula", approx(lot, expected_lot, 0.005),
      f"got {lot}, expected {expected_lot:.4f}")

# Max cap: very tight SL -> lot would be huge -> must be capped
lot_tight = calculate_position_size(BALANCE, ENTRY_LONG, ENTRY_LONG - 0.5, PARAMS)
check("Tight SL lot capped at max_lot_size", lot_tight <= PARAMS.max_lot_size,
      f"got {lot_tight}")

# Min cap: huge SL -> lot would be tiny -> at least min_lot
lot_wide = calculate_position_size(100.0, ENTRY_LONG, ENTRY_LONG - 200.0, PARAMS)
check("Huge SL lot >= min_lot_size", lot_wide >= PARAMS.min_lot_size,
      f"got {lot_wide}")

# risk_pct override
lot_override = calculate_position_size(BALANCE, ENTRY_LONG, sl_ps, PARAMS, risk_pct=0.005)
check("risk_pct override increases lot size", lot_override > lot, f"override={lot_override}, base={lot}")

# max_risk_pct hard cap
lot_capped = calculate_position_size(BALANCE, ENTRY_LONG, sl_ps, PARAMS, risk_pct=0.10)
expected_capped = (BALANCE * PARAMS.max_risk_pct) / ((abs(ENTRY_LONG - sl_ps) / PARAMS.pip_size) * PARAMS.point_value)
check("risk_pct hard cap applied",
      approx(lot_capped, min(expected_capped, PARAMS.max_lot_size), 0.005),
      f"got {lot_capped}")

# Edge cases
check("Zero balance returns None",   calculate_position_size(0, ENTRY_LONG, sl_ps) is None)
check("SL == entry returns None",    calculate_position_size(BALANCE, ENTRY_LONG, ENTRY_LONG) is None)


# ================================================================================
# TEST 5: build_trade_parameters — end to end
# ================================================================================
print(f"\n{BOLD}== TEST 5: build_trade_parameters =={RESET}")

trade = build_trade_parameters(
    signal="long", entry=ENTRY_LONG,
    fvg=FVG_BULL, displacement=DISP_BULL,
    atr=ATR, account_balance=BALANCE,
    params=PARAMS, timestamp=pd.Timestamp("2025-01-06 09:00"),
)

check("Trade is a dict",         isinstance(trade, dict))
check("Has all required keys",   all(k in trade for k in [
    "signal","entry","stop_loss","tp1","tp2","tp3",
    "risk","lot_size","risk_amount_usd","tp1_pct","tp2_pct","tp3_pct",
]))
check("signal == 'long'",        trade["signal"] == "long")
check("SL < entry",              trade["stop_loss"] < trade["entry"])
check("TP1 < TP2 < TP3",        trade["tp1"] < trade["tp2"] < trade["tp3"])
check("lot_size > 0",            trade["lot_size"] > 0)
check("risk_amount_usd > 0",     trade["risk_amount_usd"] > 0)
check("risk_amount_usd <= 2%",   trade["risk_amount_usd"] <= BALANCE * PARAMS.max_risk_pct * 10 + 1)
check("risk_reward_tp2 == 2.5",  approx(trade["risk_reward_tp2"], PARAMS.tp2_r))
check("timestamp stored",        trade["timestamp"] == pd.Timestamp("2025-01-06 09:00"))

summarise_trade(trade)

# Short trade
trade_s = build_trade_parameters(
    signal="short", entry=ENTRY_SHORT,
    fvg=FVG_BEAR, displacement=DISP_BEAR,
    atr=ATR, account_balance=BALANCE, params=PARAMS,
)
check("Short trade built",     isinstance(trade_s, dict))
check("Short SL > entry",      trade_s["stop_loss"] > trade_s["entry"])
check("Short TP1 < entry",     trade_s["tp1"] < trade_s["entry"])

# Invalid inputs
check("Invalid signal -> None", build_trade_parameters("flat", ENTRY_LONG, None, None, ATR, BALANCE) is None)


# ================================================================================
# TEST 6: validate_trade
# ================================================================================
print(f"\n{BOLD}== TEST 6: validate_trade =={RESET}")

# Valid long trade
valid, msg = validate_trade("long", 2520.0, 2505.0, 0.10, 15.0, PARAMS)
check("Valid long trade passes", valid, msg)

# Valid short trade
valid, msg = validate_trade("short", 2520.0, 2540.0, 0.10, 20.0, PARAMS)
check("Valid short trade passes", valid, msg)

# SL on wrong side
valid, msg = validate_trade("long", 2520.0, 2530.0, 0.10, 10.0, PARAMS)
check("Long SL above entry -> invalid", not valid, f"should fail but got: {msg}")

valid, msg = validate_trade("short", 2520.0, 2510.0, 0.10, 10.0, PARAMS)
check("Short SL below entry -> invalid", not valid)

# SL too tight
valid, msg = validate_trade("long", 2520.0, 2518.0, 0.10, 2.0, PARAMS)
check("SL too tight -> invalid", not valid, f"risk=2.0 < min={PARAMS.sl_min_distance}")

# SL too wide
valid, msg = validate_trade("long", 2520.0, 2460.0, 0.10, 60.0, PARAMS)
check("SL too wide -> invalid", not valid, f"risk=60 > max={PARAMS.sl_max_distance}")

# Lot out of range
valid, msg = validate_trade("long", 2520.0, 2505.0, 0.001, 15.0, PARAMS)
check("Lot below min -> invalid", not valid)

valid, msg = validate_trade("long", 2520.0, 2505.0, 100.0, 15.0, PARAMS)
check("Lot above max -> invalid", not valid)


# ================================================================================
# TEST 7: DailyRiskGuard
# ================================================================================
print(f"\n{BOLD}== TEST 7: DailyRiskGuard =={RESET}")

guard = DailyRiskGuard(
    starting_balance=BALANCE,
    max_daily_loss_pct=0.02,
    max_weekly_loss_pct=0.05,
    max_consecutive_losses=3,
)

ts = pd.Timestamp("2025-01-06 09:00")
can, msg = guard.can_trade(BALANCE, ts)
check("Fresh guard allows trading", can, msg)

# Record a winning trade
guard.record_trade(+50.0, ts)
can, msg = guard.can_trade(BALANCE + 50, ts + pd.Timedelta(hours=1))
check("After win, still can trade", can, msg)

# Record 3 consecutive losses
guard.record_trade(-30.0, ts + pd.Timedelta(hours=2))
guard.record_trade(-30.0, ts + pd.Timedelta(hours=3))
guard.record_trade(-30.0, ts + pd.Timedelta(hours=4))
# After 3 losses, should be paused
can, msg = guard.can_trade(BALANCE, ts + pd.Timedelta(hours=4, minutes=30))
check("Paused after 3 consecutive losses", not can, msg)

# After 2 hours, pause lifts
can, msg = guard.can_trade(BALANCE, ts + pd.Timedelta(hours=7))
check("Pause lifted after 2 hours", can, msg)

# Daily loss limit: lose 2% of $10,000 = $200
guard2 = DailyRiskGuard(BALANCE)
guard2.record_trade(-210.0, ts)
can2, msg2 = guard2.can_trade(BALANCE - 210, ts + pd.Timedelta(minutes=30))
check("Daily loss limit hit -> cannot trade", not can2, msg2)

# New day resets daily P&L
guard2.new_day(BALANCE - 210)
can3, msg3 = guard2.can_trade(BALANCE - 210, ts + pd.Timedelta(hours=25))
check("New day resets daily stop", can3, msg3)

# Weekly loss limit: lose 5% of $10,000 = $500
guard3 = DailyRiskGuard(BALANCE)
for _ in range(5):
    guard3.record_trade(-110.0, ts)
    guard3.new_day(BALANCE)   # reset daily each day
can4, msg4 = guard3.can_trade(BALANCE, ts + pd.Timedelta(days=5))
check("Weekly loss limit hit -> cannot trade", not can4, msg4)

# Summary
print()
print("  Guard summary:")
for k, v in guard.summary().items():
    print(f"    {k}: {v}")


# ================================================================================
# TEST 8: ScalpRiskParams customisation
# ================================================================================
print(f"\n{BOLD}== TEST 8: ScalpRiskParams Customisation =={RESET}")

custom = ScalpRiskParams(risk_pct=0.005, tp1_r=2.0, tp2_r=3.0, tp3_r=5.0)
check("Custom risk_pct stored",  custom.risk_pct == 0.005)
check("Custom tp1_r stored",     custom.tp1_r == 2.0)

tps_custom = calculate_take_profits("long", 2520.0, 2505.0, custom)
check("TP1 uses custom tp1_r",  approx(tps_custom["tp1"], 2520 + 2.0 * 15))
check("TP2 uses custom tp2_r",  approx(tps_custom["tp2"], 2520 + 3.0 * 15))

lot_custom = calculate_position_size(BALANCE, ENTRY_LONG, 2505.0, custom)
lot_default = calculate_position_size(BALANCE, ENTRY_LONG, 2505.0, PARAMS)
check("Higher risk_pct -> larger lot", lot_custom > lot_default,
      f"custom={lot_custom}, default={lot_default}")


# ================================================================================
# SUMMARY
# ================================================================================
print(f"\n{BOLD}{'='*55}{RESET}")
total = passed + failed
if failed == 0:
    print(f"{BOLD}{GREEN}  ALL {total} TESTS PASSED{RESET}")
    print(f"{GREEN}  risk_model.py is ready.{RESET}")
else:
    print(f"{BOLD}{RED}  {failed}/{total} TESTS FAILED{RESET}")
print(f"{BOLD}{'='*55}{RESET}\n")

sys.exit(0 if failed == 0 else 1)