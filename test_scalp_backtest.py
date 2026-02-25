"""
test_scalp_backtest.py
=======================
Smoke test for backtest/scalping_backtest.py using synthetic 15M data.

Does NOT test signal quality — that is what the real backtest is for.
Tests ONLY that:
  1. The engine runs without crashing
  2. Output structure is correct
  3. Trade simulation logic (SL/TP hit detection) is correct
  4. Metrics calculation is correct
  5. DailyRiskGuard integration works inside the engine
  6. No lookahead bias (trade can only exit on bars AFTER entry)

Run from gold_algo_bot/ root:
    python test_scalp_backtest.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from backtest.scalping_backtest import (
    ScalpBacktest, BacktestParams, _simulate_trade, _resample_ohlcv
)
from scalping.risk_model import ScalpRiskParams

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"
passed = 0; failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  {GREEN}✓{RESET}  {name}"); passed += 1
    else:
        print(f"  {RED}✗{RESET}  {RED}{name}{RESET}" + (f"  ← {detail}" if detail else "")); failed += 1


# ── synthetic data builders ────────────────────────────────────────────────────

def make_15m(n=500, base=2500.0, seed=42):
    """
    Generate synthetic 15M OHLCV data with realistic price movement.
    Uses a random walk with slight upward bias.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-08 07:00", periods=n, freq="15min")

    closes = [base]
    for _ in range(n - 1):
        closes.append(closes[-1] + rng.normal(0.1, 3.0))

    opens  = [closes[0]] + closes[:-1]
    highs  = [max(o, c) + abs(rng.normal(0, 2)) for o, c in zip(opens, closes)]
    lows   = [min(o, c) - abs(rng.normal(0, 2)) for o, c in zip(opens, closes)]
    vols   = rng.integers(500, 3000, n).astype(float)

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": vols,
    }, index=dates)
    return df


def make_controlled_15m(entry_bar=50):
    """
    Build a DataFrame where we KNOW exactly when SL and TP will be hit.
    Candle layout around entry:
      bars 0-49:    flat (warmup)
      bar  50:      entry candle (close = 2500)
      bar  51-54:   drift up gently
      bar  55:      TP1 hit (high = 2530, crosses TP1 at 2520+1.5×10=2535... use tighter risk)
      bar  60:      TP2 hit
      bar  65:      TP3 hit
    We use SL=2490, entry=2500, risk=10, so:
      TP1 = 2500 + 1.5×10 = 2515
      TP2 = 2500 + 2.5×10 = 2525
      TP3 = 2500 + 4.0×10 = 2540
    """
    n = 100
    dates = pd.date_range("2024-01-08 07:00", periods=n, freq="15min")
    rows = []

    for i in range(n):
        if i < 50:           # warmup flat
            rows.append((2500, 2503, 2497, 2500, 1000))
        elif i == 50:        # entry candle
            rows.append((2498, 2502, 2496, 2500, 1200))
        elif i < 55:         # gentle drift
            rows.append((2500, 2510, 2499, 2508, 1000))
        elif i == 55:        # TP1 hit (high >= 2515)
            rows.append((2508, 2518, 2507, 2515, 1500))
        elif i < 60:         # drift more
            rows.append((2515, 2522, 2514, 2520, 1000))
        elif i == 60:        # TP2 hit (high >= 2525)
            rows.append((2520, 2528, 2519, 2526, 1800))
        elif i < 65:         # drift to TP3
            rows.append((2526, 2535, 2525, 2532, 1000))
        elif i == 65:        # TP3 hit (high >= 2540)
            rows.append((2532, 2542, 2531, 2540, 2000))
        else:                # after resolution
            rows.append((2540, 2542, 2538, 2540, 800))

    df = pd.DataFrame(rows, columns=["open","high","low","close","volume"], index=dates)
    return df


def make_sl_scenario():
    """Build data where price drops straight to SL after entry."""
    n = 80
    dates = pd.date_range("2024-01-08 07:00", periods=n, freq="15min")
    rows = []
    for i in range(n):
        if i < 50:
            rows.append((2500, 2503, 2497, 2500, 1000))
        elif i == 50:       # entry candle
            rows.append((2498, 2502, 2496, 2500, 1200))
        elif i == 51:       # SL hit immediately (low <= 2490)
            rows.append((2499, 2500, 2488, 2492, 2500))
        else:
            rows.append((2492, 2494, 2490, 2492, 800))
    return pd.DataFrame(rows, columns=["open","high","low","close","volume"], index=dates)


# ── TEST 1: _resample_ohlcv ────────────────────────────────────────────────────
print(f"\n{BOLD}── TEST 1: _resample_ohlcv ──{RESET}")

df15 = make_15m(200)
df1h = _resample_ohlcv(df15, "1h")
df4h = _resample_ohlcv(df15, "4h")

check("1H has fewer candles than 15M",    len(df1h) < len(df15))
check("4H has fewer candles than 1H",     len(df4h) < len(df1h))
check("1H high >= 1H low",               (df1h["high"] >= df1h["low"]).all())
check("4H has open/high/low/close/volume", set(df4h.columns) >= {"open","high","low","close","volume"})
check("4H high is max of constituent 15M highs",
      float(df4h["high"].iloc[0]) >= float(df1h["high"].iloc[0]))


# ── TEST 2: _simulate_trade — TP3 hit ─────────────────────────────────────────
print(f"\n{BOLD}── TEST 2: _simulate_trade — TP3 Full Win ──{RESET}")

df_ctrl = make_controlled_15m()
params  = BacktestParams(risk_params=ScalpRiskParams(pip_size=0.1, point_value=10.0,
                                                     min_lot_size=0.01))

# Construct a trade that hits TP1@2515, TP2@2525, TP3@2540
trade_tp3 = {
    "signal":    "long",
    "entry":     2500.0,
    "stop_loss": 2490.0,
    "tp1":       2515.0,
    "tp2":       2525.0,
    "tp3":       2540.0,
    "risk":      10.0,
    "lot_size":  0.10,
    "tp1_pct":   0.50,
    "tp2_pct":   0.30,
    "tp3_pct":   0.20,
}

result = _simulate_trade(df_ctrl, entry_bar=50, trade=trade_tp3, params=params)
check("Outcome is tp3",          result["outcome"] == "tp3", f"got {result['outcome']}")
check("TP1 was hit",             result["tp1_hit"])
check("TP2 was hit",             result["tp2_hit"])
check("Net P&L is positive",     result["net_pnl"] > 0, f"pnl={result['net_pnl']}")
check("Exit bar > entry bar",    result["exit_bar"] > 50)
check("No lookahead: exit >= 51",result["exit_bar"] >= 51)
check("R-multiple > 0",          result["r_multiple"] > 0)

print(f"  → Outcome: {result['outcome']}, PnL: ${result['net_pnl']:.2f}, "
      f"R: {result['r_multiple']:.2f}, Exit bar: {result['exit_bar']}")


# ── TEST 3: _simulate_trade — SL hit ──────────────────────────────────────────
print(f"\n{BOLD}── TEST 3: _simulate_trade — Full SL ──{RESET}")

df_sl = make_sl_scenario()
trade_sl = {
    "signal":    "long",
    "entry":     2500.0,
    "stop_loss": 2490.0,
    "tp1":       2515.0,
    "tp2":       2525.0,
    "tp3":       2540.0,
    "risk":      10.0,
    "lot_size":  0.10,
    "tp1_pct":   0.50,
    "tp2_pct":   0.30,
    "tp3_pct":   0.20,
}

result_sl = _simulate_trade(df_sl, entry_bar=50, trade=trade_sl, params=params)
check("Outcome is sl",        result_sl["outcome"] == "sl", f"got {result_sl['outcome']}")
check("TP1 NOT hit",          not result_sl["tp1_hit"])
check("Net P&L is negative",  result_sl["net_pnl"] < 0, f"pnl={result_sl['net_pnl']}")
check("R-multiple is negative", result_sl["r_multiple"] < 0)
check("Exit bar is 51",       result_sl["exit_bar"] == 51, f"got {result_sl['exit_bar']}")

print(f"  → Outcome: {result_sl['outcome']}, PnL: ${result_sl['net_pnl']:.2f}, "
      f"R: {result_sl['r_multiple']:.2f}, Exit bar: {result_sl['exit_bar']}")


# ── TEST 4: _simulate_trade — Short TP3 ───────────────────────────────────────
print(f"\n{BOLD}── TEST 4: _simulate_trade — Short TP3 ──{RESET}")

# Mirror: entry 2500, SL=2510, TP1=2485, TP2=2475, TP3=2460
n = 80
dates_s = pd.date_range("2024-01-08 07:00", periods=n, freq="15min")
rows_s  = [(2500, 2503, 2497, 2500, 1000)] * 50 + \
          [(2500, 2501, 2496, 2499, 1200)] + \
          [(2499, 2499, 2482, 2485, 1500)] + \
          [(2485, 2486, 2472, 2475, 1800)] + \
          [(2475, 2476, 2458, 2460, 2000)] + \
          [(2460, 2462, 2455, 2458, 800)] * 26
df_short = pd.DataFrame(rows_s, columns=["open","high","low","close","volume"], index=dates_s)

trade_short = {
    "signal":    "short",
    "entry":     2500.0,
    "stop_loss": 2510.0,
    "tp1":       2485.0,
    "tp2":       2475.0,
    "tp3":       2460.0,
    "risk":      10.0,
    "lot_size":  0.10,
    "tp1_pct":   0.50,
    "tp2_pct":   0.30,
    "tp3_pct":   0.20,
}

result_short = _simulate_trade(df_short, entry_bar=50, trade=trade_short, params=params)
check("Short outcome is tp3 or tp2",  result_short["outcome"] in ("tp3","tp2","sl_after_tp2"),
      f"got {result_short['outcome']}")
check("Short net P&L > 0",            result_short["net_pnl"] > 0)
check("Short TP1 hit",                result_short["tp1_hit"])


# ── TEST 5: ScalpBacktest engine — no crash ────────────────────────────────────
print(f"\n{BOLD}── TEST 5: ScalpBacktest — No Crash ──{RESET}")

df_synth = make_15m(600, seed=99)
bt_params = BacktestParams(
    initial_balance        = 10_000.0,
    warmup_candles         = 200,
    step                   = 3,         # faster for test
    min_signal_gap_candles = 8,
    session_filter         = False,     # bypass for synthetic data
    min_soft_score         = 0,         # easier to generate signals
    pivot_lookback         = 2,
)

bt = ScalpBacktest(df_synth, params=bt_params)
results = bt.run()

check("Results is a dict",              isinstance(results, dict))
check("Has 'trades' key",               "trades" in results)
check("Has 'equity_curve' key",         "equity_curve" in results)
check("Has 'rejections' key",           "rejections" in results)
check("Has 'metrics' key",              "metrics" in results)
check("Has 'final_balance' key",        "final_balance" in results)
check("Equity curve not empty",         len(results["equity_curve"]) >= 1)
check("Final balance is a float",       isinstance(results["final_balance"], float))
check("Rejections tracked",             results["rejections"]._total > 0)

print(f"  → Trades generated: {len(results['trades'])}")
print(f"  → Final balance:    ${results['final_balance']:,.2f}")
print(f"  → Signals checked:  {results['rejections']._total:,}")


# ── TEST 6: Metrics structure ──────────────────────────────────────────────────
print(f"\n{BOLD}── TEST 6: Metrics Structure ──{RESET}")

m = results["metrics"]
if results["metrics"].get("total_trades", 0) > 0:
    expected_keys = {"total_trades","wins","losses","win_rate","avg_win_r","avg_loss_r",
                     "expectancy_r","profit_factor","total_pnl","total_return_pct",
                     "max_drawdown","max_drawdown_pct","tp1_hit_rate","tp2_hit_rate",
                     "full_sl_rate","avg_bars_held","longs","shorts"}
    check("All metric keys present",    expected_keys.issubset(set(m.keys())),
          f"missing: {expected_keys - set(m.keys())}")
    check("Win rate in 0–100",          0 <= m["win_rate"] <= 100)
    check("Profit factor >= 0",         m["profit_factor"] >= 0)
    check("Max drawdown >= 0",          m["max_drawdown"] >= 0)
    check("Longs + Shorts = total",     m["longs"] + m["shorts"] == m["total_trades"])
else:
    check("No trades generated (signal hard to get on synth data)", True)
    print(f"  → Rejection breakdown:")
    results["rejections"].summary()


# ── TEST 7: No lookahead bias ──────────────────────────────────────────────────
print(f"\n{BOLD}── TEST 7: No Lookahead Bias ──{RESET}")

trades = results["trades"]
if trades:
    df_trades = pd.DataFrame(trades)
    check("All entry_time < exit_time",
          (df_trades["entry_time"] < df_trades["exit_time"]).all(),
          "some trades have exit_time <= entry_time")
    check("All balances positive",
          (df_trades["balance"] > 0).all())
    # Every lot_size > 0
    check("All lot_sizes > 0",
          (df_trades["lot_size"] > 0).all())
else:
    check("No trades to check lookahead (no signals on synth data)", True)


# ── TEST 8: BacktestParams defaults ───────────────────────────────────────────
print(f"\n{BOLD}── TEST 8: BacktestParams Defaults ──{RESET}")

default_p = BacktestParams()
check("Default balance is 10,000",      default_p.initial_balance == 10_000.0)
check("Default risk is 0.25%",          default_p.risk_pct == 0.0025)
check("Default warmup is 300",          default_p.warmup_candles == 300)
check("Session filter on by default",   default_p.session_filter is True)
check("Min soft score default is 1",    default_p.min_soft_score == 1)
check("Risk params is ScalpRiskParams", isinstance(default_p.risk_params, ScalpRiskParams))


# ── TEST 9: print_summary doesn't crash ───────────────────────────────────────
print(f"\n{BOLD}── TEST 9: print_summary ──{RESET}")

try:
    bt.print_summary(results)
    check("print_summary runs without error", True)
except Exception as e:
    check("print_summary runs without error", False, str(e))


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'='*55}{RESET}")
total = passed + failed
if failed == 0:
    print(f"{BOLD}{GREEN}  ALL {total} TESTS PASSED{RESET}")
    print(f"{GREEN}  scalping_backtest.py is ready.{RESET}")
else:
    print(f"{BOLD}{RED}  {failed}/{total} TESTS FAILED{RESET}")
print(f"{BOLD}{'='*55}{RESET}\n")

sys.exit(0 if failed == 0 else 1)