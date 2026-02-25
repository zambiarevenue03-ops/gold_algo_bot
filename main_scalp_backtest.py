"""
main_scalp_backtest.py
=======================
Runner script for the XAUUSD scalping bot backtest.

Usage:
    python main_scalp_backtest.py

Reads your real CSV data, runs the full backtest across all regimes,
prints results, and saves the trade journal + equity curve.
"""

import sys
import os
import pandas as pd

# ── path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.data_loader         import load_csv
from backtest.scalping_backtest import ScalpBacktest, BacktestParams
from scalping.risk_model        import ScalpRiskParams

# ── data paths ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT, "data", "processed")

FILES = {
    "15m": os.path.join(DATA_DIR, "XAUUSD_15M.csv"),
    "1h":  os.path.join(DATA_DIR, "XAUUSD_1H.csv"),
    "4h":  os.path.join(DATA_DIR, "XAUUSD_4H.csv"),
}

OUTPUT_DIR = os.path.join(ROOT, "backtest", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── regime definitions ─────────────────────────────────────────────────────────
# Dates reference the 15M data range: 2023-02-10 → 2026-02-10
REGIMES = [
    {
        "name":  "Full History",
        "start": "2023-02-10",
        "end":   "2026-02-10",
        "note":  "All available data",
    },
    {
        "name":  "Bull Run",
        "start": "2023-10-01",
        "end":   "2024-05-01",
        "note":  "1820→2400 breakout",
    },
    {
        "name":  "ATH Push",
        "start": "2024-08-01",
        "end":   "2025-01-01",
        "note":  "2400→2700+ all-time highs",
    },
    {
        "name":  "Recent",
        "start": "2025-01-01",
        "end":   "2026-02-10",
        "note":  "Most recent 13 months",
    },
]


# ── backtest parameters ────────────────────────────────────────────────────────
PARAMS = BacktestParams(
    initial_balance        = 10_000.0,
    risk_pct               = 0.0025,       # 0.25% per trade
    warmup_candles         = 300,          # ~3 days of 15M candles
    step                   = 1,            # check every candle
    min_signal_gap_candles = 16,           # minimum 4 hours between entries
    session_filter         = True,
    utc_offset_hours       = 0,            # adjust if your CSV is not UTC
    min_soft_score         = 1,            # at least 1 soft confirmation
    pivot_lookback         = 2,
    obv_ma_period          = 10,
    atr_period             = 14,
    atr_ma_period          = 14,
    fvg_lookback           = 50,
    disp_lookback          = 30,
    choch_max_candles      = 15,
    risk_params            = ScalpRiskParams(
        risk_pct       = 0.0025,
        sl_buffer_atr  = 0.3,
        tp1_r          = 1.5,
        tp2_r          = 2.5,
        tp3_r          = 4.0,
        sl_max_distance = 50.0,
    ),
)


# ── helpers ────────────────────────────────────────────────────────────────────

def load_data():
    """Load and validate all required CSV files."""
    print("\n" + "═"*55)
    print("  LOADING DATA")
    print("═"*55)

    missing = [k for k, v in FILES.items() if not os.path.exists(v)]
    if missing:
        print(f"\n  ERROR: Missing files: {missing}")
        print(f"  Expected in: {DATA_DIR}")
        sys.exit(1)

    df_15m = load_csv(FILES["15m"])
    print(f"  15M: {len(df_15m):>8,} candles  "
          f"({df_15m.index[0].date()} → {df_15m.index[-1].date()})")

    return df_15m


def run_regime(df_15m: pd.DataFrame, regime: dict, params: BacktestParams) -> dict:
    """Run backtest for a single date range."""
    start = pd.Timestamp(regime["start"])
    end   = pd.Timestamp(regime["end"])

    slice_df = df_15m[(df_15m.index >= start) & (df_15m.index <= end)].copy()

    if len(slice_df) < params.warmup_candles + 100:
        print(f"\n  SKIP {regime['name']}: only {len(slice_df)} candles")
        return None

    print(f"\n{'─'*55}")
    print(f"  REGIME: {regime['name']}  ({regime['note']})")
    print(f"  Range: {start.date()} → {end.date()}  ({len(slice_df):,} candles)")
    print(f"{'─'*55}")

    bt      = ScalpBacktest(slice_df, params)
    results = bt.run()
    bt.print_summary(results)

    # Save outputs
    safe_name = regime["name"].lower().replace(" ", "_")
    journal_path  = os.path.join(OUTPUT_DIR, f"scalp_journal_{safe_name}.csv")
    equity_path   = os.path.join(OUTPUT_DIR, f"scalp_equity_{safe_name}.csv")
    bt.save_journal(results, journal_path)
    bt.save_equity_curve(results, equity_path)

    return results


def print_regime_comparison(all_results: list):
    """Print a side-by-side comparison table of all regimes."""
    print(f"\n{'═'*70}")
    print(f"  REGIME COMPARISON")
    print(f"{'═'*70}")
    print(f"  {'Regime':<18} {'Trades':>7} {'WR%':>6} {'Exp R':>7} "
          f"{'PF':>5} {'Return%':>8} {'MaxDD%':>7}")
    print(f"  {'─'*68}")

    for r in all_results:
        if r is None:
            continue
        name    = r["regime_name"]
        m       = r["results"]["metrics"]
        if m.get("total_trades", 0) == 0:
            print(f"  {name:<18} {'No trades':>50}")
            continue
        print(f"  {name:<18} {m['total_trades']:>7} {m['win_rate']:>6.1f} "
              f"{m['expectancy_r']:>+7.3f} {m['profit_factor']:>5.2f} "
              f"{m['total_return_pct']:>+8.2f} {m['max_drawdown_pct']:>7.2f}")

    print(f"{'═'*70}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*55)
    print("  XAUUSD SCALPING BOT — BACKTEST")
    print("═"*55)

    df_15m = load_data()
    all_results = []

    for regime in REGIMES:
        results = run_regime(df_15m, regime, PARAMS)
        if results is not None:
            all_results.append({"regime_name": regime["name"], "results": results})

    if len(all_results) > 1:
        print_regime_comparison(all_results)

    print(f"\n  Output files saved to: {OUTPUT_DIR}")
    print(f"  Open scalp_journal_*.csv to inspect individual trades.\n")


if __name__ == "__main__":
    main()



    