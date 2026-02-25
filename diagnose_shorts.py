"""
diagnose_shorts.py
===================
Debug tool to find out why no short signals are being generated.

Checks every layer of the entry logic to see where shorts are being filtered out.

Usage:
    python diagnose_shorts.py
"""

import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pandas as pd
from utils.data_loader import load_csv
from scalping.entry_logic import check_entry_signal_detailed
from indicators.pivots import detect_swings
from smc.structure import determine_trend

# ── Load data ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT, "data", "processed")
df_15m = load_csv(os.path.join(DATA_DIR, "XAUUSD_15M.csv"))

print(f"\n{'═'*70}")
print(f"  SHORTS DIAGNOSTIC TOOL")
print(f"{'═'*70}")
print(f"  Loaded {len(df_15m):,} candles of 15M data")
print(f"  Period: {df_15m.index[0].date()} → {df_15m.index[-1].date()}")
print(f"\n  Checking why short signals are not generating...")
print(f"{'═'*70}\n")

# ── Helper to resample ─────────────────────────────────────────────────────────
def resample_ohlcv(df, rule):
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna()

# ── Sample points to check ─────────────────────────────────────────────────────
# Check 1000 random points across the dataset
sample_size = 1000
step = len(df_15m) // sample_size

bearish_bias_count = 0
bearish_bias_samples = []

print(f"  Sampling {sample_size} points to check HTF bias...\n")

for i in range(300, len(df_15m), step):
    slice_15m = df_15m.iloc[:i + 1]
    
    # Resample
    df_1h = resample_ohlcv(slice_15m, "1h")
    df_4h = resample_ohlcv(slice_15m, "4h")
    
    if len(df_1h) < 60 or len(df_4h) < 30:
        continue
    
    # Check trend on 4H
    sh_4h, sl_4h = detect_swings(df_4h, lookback=2)
    trend_4h = determine_trend(sh_4h, sl_4h)
    
    # Check trend on 1H
    sh_1h, sl_1h = detect_swings(df_1h, lookback=2)
    trend_1h = determine_trend(sh_1h, sl_1h)
    
    # Check if both bearish
    if trend_4h == "bearish" and trend_1h == "bearish":
        bearish_bias_count += 1
        if len(bearish_bias_samples) < 10:
            bearish_bias_samples.append({
                "timestamp": df_15m.index[i],
                "trend_4h": trend_4h,
                "trend_1h": trend_1h,
            })

print(f"{'─'*70}")
print(f"  BIAS DISTRIBUTION (sampled {sample_size} points)")
print(f"{'─'*70}")
print(f"  Bearish bias detected:  {bearish_bias_count}/{sample_size} ({bearish_bias_count/sample_size*100:.1f}%)")
print(f"  Bullish bias expected:  ~{sample_size - bearish_bias_count}/{sample_size}")

if bearish_bias_count == 0:
    print(f"\n  🚨 PROBLEM FOUND:")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  NO bearish bias detected in ANY of the {sample_size} samples.")
    print(f"  This means 4H and 1H trends are NEVER both bearish together.")
    print(f"\n  LIKELY CAUSES:")
    print(f"    1. Gold has been in a strong bull market since 2023")
    print(f"       → 4H and 1H rarely align bearish during this period")
    print(f"    2. Your swing detection requires BOTH timeframes bearish")
    print(f"       → This is extremely conservative for shorts")
    print(f"    3. determine_trend() may be biased toward bullish")
    print(f"\n  SOLUTIONS:")
    print(f"    A. Relax HTF requirement: Allow shorts when EITHER 4H OR 1H is bearish")
    print(f"       (instead of requiring both)")
    print(f"    B. Add a separate bearish entry mode that triggers on:")
    print(f"       - Price below 50MA on 4H")
    print(f"       - Recent lower high on 1H")
    print(f"    C. Accept that this bull market doesn't support shorts")
    print(f"       → Stay long-only until market regime changes")
    print(f"  ─────────────────────────────────────────────────────────")

elif bearish_bias_count < sample_size * 0.1:
    print(f"\n  ⚠️  ISSUE DETECTED:")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Bearish bias is RARE ({bearish_bias_count/sample_size*100:.1f}% of samples).")
    print(f"  Gold has been in a strong bull trend since 2023.")
    print(f"\n  When bearish bias DID occur:")
    for sample in bearish_bias_samples[:5]:
        print(f"    {sample['timestamp'].date()}  (4H: {sample['trend_4h']}, 1H: {sample['trend_1h']})")
    print(f"\n  NEXT STEP:")
    print(f"  Check if SHORT signals are generated during these periods...")

    # Run full entry check on one bearish sample
    if bearish_bias_samples:
        test_point = bearish_bias_samples[0]
        idx = df_15m.index.get_loc(test_point["timestamp"])
        
        slice_15m = df_15m.iloc[:idx + 1]
        df_1h_test = resample_ohlcv(slice_15m, "1h")
        df_4h_test = resample_ohlcv(slice_15m, "4h")
        
        signal, reason, scores = check_entry_signal_detailed(
            df_1h_test, df_4h_test, slice_15m,
            session_filter=False,  # bypass session for diagnostics
            min_soft_score=0,      # bypass soft score
        )
        
        print(f"\n  Testing entry logic at {test_point['timestamp'].date()}:")
        print(f"    Signal:      {signal}")
        print(f"    Reason:      {reason}")
        print(f"    Bias:        {scores.get('bias')}")
        print(f"    OBV aligned: {scores.get('obv_aligned')}")
        print(f"    ATR expanding: {scores.get('atr_expanding')}")
        print(f"    Displacement: {scores.get('displacement') is not None}")
        print(f"    FVG:         {scores.get('fvg') is not None}")
        
        if signal is None:
            print(f"\n  Even with bearish bias, signal was blocked by: {reason}")

else:
    print(f"\n  ✅ Bearish bias is present in {bearish_bias_count/sample_size*100:.1f}% of samples.")
    print(f"  This is healthy. Checking why shorts aren't firing...\n")
    
    # Test a bearish sample point
    test_point = bearish_bias_samples[0]
    idx = df_15m.index.get_loc(test_point["timestamp"])
    
    slice_15m = df_15m.iloc[:idx + 1]
    df_1h_test = resample_ohlcv(slice_15m, "1h")
    df_4h_test = resample_ohlcv(slice_15m, "4h")
    
    signal, reason, scores = check_entry_signal_detailed(
        df_1h_test, df_4h_test, slice_15m,
        session_filter=False,
        min_soft_score=0,
    )
    
    print(f"  Testing entry logic at {test_point['timestamp'].date()}:")
    print(f"    Signal:      {signal}")
    print(f"    Reason:      {reason}")
    print(f"    Scores:      {scores}")

print(f"\n{'═'*70}\n")