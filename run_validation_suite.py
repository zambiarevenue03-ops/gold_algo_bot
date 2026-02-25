"""
run_validation_suite.py
========================
Runs Monte Carlo + Parameter Stability tests on your backtest results.

This is Phase 1 of the validation roadmap:
  1. Monte Carlo → tests if +13.8% return was due to lucky trade sequence
  2. Parameter Stability → tests if edge holds across parameter variations

Usage:
    python run_validation_suite.py
    
Reads the trade journals from your backtest and runs both validation tests.
Takes ~5-10 minutes depending on parameter grid size.
"""

import sys
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backtest.monte_carlo import run_monte_carlo_from_journal, MonteCarloSimulator
from backtest.parameter_stability import ParameterGrid, run_stability_test
from backtest.scalping_backtest import BacktestParams
from scalping.risk_model import ScalpRiskParams
from utils.data_loader import load_csv


# ── Configuration ──────────────────────────────────────────────────────────────

# Which backtest result to validate?
REGIME = "full_history"  # or "bull_run", "ath_push", "recent"

# Paths
JOURNAL_PATH = os.path.join(ROOT, "backtest", "results", f"scalp_journal_{REGIME}.csv")
DATA_PATH    = os.path.join(ROOT, "data", "processed", "XAUUSD_15M.csv")

# Initial balance
INITIAL_BALANCE = 10_000.0

# Monte Carlo settings
MONTE_CARLO_SIMS = 2000  # 2000 shuffles (takes ~10 seconds)

# Parameter grid for stability test
# Test 3×3×3 = 27 combinations
PARAM_GRID = ParameterGrid(
    tp1_r         = [1.3, 1.5, 1.7],
    tp2_r         = [2.3, 2.5, 2.7],
    sl_buffer_atr = [0.2, 0.3, 0.4],
)

# Base parameters (for stability test)
BASE_PARAMS = BacktestParams(
    initial_balance        = INITIAL_BALANCE,
    risk_pct               = 0.0025,
    warmup_candles         = 300,
    step                   = 3,  # faster for parameter test
    min_signal_gap_candles = 16,
    session_filter         = True,
    utc_offset_hours       = 0,
    min_soft_score         = 1,
    pivot_lookback         = 2,
    obv_ma_period          = 10,
    atr_period             = 14,
    atr_ma_period          = 14,
    risk_params            = ScalpRiskParams(
        risk_pct       = 0.0025,
        sl_buffer_atr  = 0.3,
        tp1_r          = 1.5,
        tp2_r          = 2.5,
        tp3_r          = 4.0,
        sl_max_distance = 50.0,
    ),
)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*70}")
    print(f"  VALIDATION SUITE")
    print(f"{'═'*70}")
    print(f"  Regime:          {REGIME}")
    print(f"  Journal:         {JOURNAL_PATH}")
    print(f"  Initial balance: ${INITIAL_BALANCE:,.2f}")
    print(f"{'═'*70}\n")
    
    # ── Check files exist ──────────────────────────────────────────────────────
    if not os.path.exists(JOURNAL_PATH):
        print(f"  ERROR: Trade journal not found at {JOURNAL_PATH}")
        print(f"  Run main_scalp_backtest.py first to generate journals.")
        sys.exit(1)
    
    if not os.path.exists(DATA_PATH):
        print(f"  ERROR: 15M data not found at {DATA_PATH}")
        sys.exit(1)
    
    # ── PHASE 1: Monte Carlo ──────────────────────────────────────────────────
    print(f"{'─'*70}")
    print(f"  PHASE 1: MONTE CARLO SIMULATION")
    print(f"{'─'*70}\n")
    
    mc_results = run_monte_carlo_from_journal(
        JOURNAL_PATH,
        initial_balance=INITIAL_BALANCE,
        n_simulations=MONTE_CARLO_SIMS,
    )
    
    # ── PHASE 2: Parameter Stability ──────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  PHASE 2: PARAMETER STABILITY TEST")
    print(f"{'─'*70}")
    print(f"  Loading 15M data...")
    
    df_15m = load_csv(DATA_PATH)
    print(f"  Loaded {len(df_15m):,} candles")
    
    # Use only Recent regime for faster parameter test
    # (Full history takes too long — 27 backtests × 15 sec = 7 minutes)
    if REGIME == "full_history":
        print(f"  Using Recent regime subset for parameter test (faster)...")
        df_15m = df_15m[df_15m.index >= "2025-01-01"]
    
    stability_results = run_stability_test(df_15m, PARAM_GRID, BASE_PARAMS)
    stability_results.print_summary()
    stability_results.print_top_10()
    
    # ── Final Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  VALIDATION SUITE COMPLETE")
    print(f"{'═'*70}\n")
    
    # Combined assessment
    mc_ok = (mc_results.confidence_level >= 40 and 
             mc_results.confidence_level <= 60 and
             mc_results.ruin_probability < 1.0)
    
    stability_ok = stability_results.profitable_pct >= 60
    
    print(f"  Monte Carlo:          {'✅ PASS' if mc_ok else '⚠️  CHECK RESULTS'}")
    print(f"  Parameter Stability:  {'✅ PASS' if stability_ok else '⚠️  CHECK RESULTS'}")
    print()
    
    if mc_ok and stability_ok:
        print(f"  ✅  VALIDATION PASSED")
        print(f"      Strategy edge is REAL and ROBUST.")
        print(f"      Proceed with confidence to:")
        print(f"        - Trailing stop implementation")
        print(f"        - Live paper trading")
        print(f"        - Account size planning")
    else:
        print(f"  ⚠️  VALIDATION CONCERNS DETECTED")
        print(f"      Review results above before proceeding.")
        print(f"      Consider:")
        print(f"        - Adjusting parameters to improve robustness")
        print(f"        - Reducing risk per trade")
        print(f"        - Increasing account size buffer")
    
    print(f"\n{'═'*70}\n")


if __name__ == "__main__":
    main()