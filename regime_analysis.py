"""
regime_analysis.py
===================
Classifies gold market into regimes and tests strategy performance in each.

Regime Types:
  1. STRONG BULL    — Price above 200MA, 50MA > 200MA, consistent higher highs
  2. WEAK BULL      — Price above 200MA but choppy, frequent pullbacks
  3. CONSOLIDATION  — Price oscillating around 200MA, no clear trend
  4. WEAK BEAR      — Price below 200MA but choppy
  5. STRONG BEAR    — Price below 200MA, 50MA < 200MA, consistent lower lows

Then runs backtest on EACH regime separately to see:
  - Which regimes are profitable
  - Which regimes to avoid
  - Optimal parameters per regime

Usage:
    python regime_analysis.py
"""

import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
from utils.data_loader import load_csv
from backtest.scalping_backtest import ScalpBacktest, BacktestParams
from scalping.risk_model import ScalpRiskParams

# ── Load Data ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT, "data", "processed")
df_15m = load_csv(os.path.join(DATA_DIR, "XAUUSD_15M.csv"))
df_daily = load_csv(os.path.join(DATA_DIR, "XAUUSD_D1.csv")) if os.path.exists(os.path.join(DATA_DIR, "XAUUSD_D1.csv")) else None

print(f"\n{'═'*70}")
print(f"  REGIME ANALYSIS")
print(f"{'═'*70}")
print(f"  Loaded {len(df_15m):,} 15M candles")
print(f"  Period: {df_15m.index[0].date()} → {df_15m.index[-1].date()}")
print(f"{'═'*70}\n")


# ── Regime Classification ──────────────────────────────────────────────────────

def classify_regime_simple(df: pd.DataFrame) -> pd.Series:
    """
    Simple regime classification using 50MA and 200MA on daily timeframe.
    
    Returns a Series with regime labels for each day.
    """
    # Resample to daily if needed
    if len(df) > 10000:  # likely 15M data
        daily = df.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
    else:
        daily = df.copy()
    
    # Calculate MAs
    daily['ma50'] = daily['close'].rolling(50).mean()
    daily['ma200'] = daily['close'].rolling(200).mean()
    
    # Calculate trend strength (ADX-like)
    daily['high_high'] = daily['high'].rolling(20).max()
    daily['low_low'] = daily['low'].rolling(20).min()
    daily['range_pct'] = (daily['high_high'] - daily['low_low']) / daily['close'] * 100
    
    # Classify
    regime = pd.Series(index=daily.index, dtype=str)
    
    for i in range(200, len(daily)):
        price = daily['close'].iloc[i]
        ma50 = daily['ma50'].iloc[i]
        ma200 = daily['ma200'].iloc[i]
        range_pct = daily['range_pct'].iloc[i]
        
        # Strong Bull: Price > MA200, MA50 > MA200, wide range
        if price > ma200 and ma50 > ma200 and range_pct > 8:
            regime.iloc[i] = "STRONG_BULL"
        
        # Weak Bull: Price > MA200 but choppy
        elif price > ma200 and ma50 > ma200:
            regime.iloc[i] = "WEAK_BULL"
        
        # Strong Bear: Price < MA200, MA50 < MA200, wide range
        elif price < ma200 and ma50 < ma200 and range_pct > 8:
            regime.iloc[i] = "STRONG_BEAR"
        
        # Weak Bear: Price < MA200 but choppy
        elif price < ma200 and ma50 < ma200:
            regime.iloc[i] = "WEAK_BEAR"
        
        # Consolidation: Price oscillating around MA200
        else:
            regime.iloc[i] = "CONSOLIDATION"
    
    return regime


def classify_regime_advanced(df: pd.DataFrame) -> pd.Series:
    """
    Advanced regime classification using ATR, volatility, and trend metrics.
    
    More sophisticated than simple MA crossover.
    """
    daily = df.resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna() if len(df) > 10000 else df.copy()
    
    # Technical indicators
    daily['ma50'] = daily['close'].rolling(50).mean()
    daily['ma200'] = daily['close'].rolling(200).mean()
    daily['atr'] = (daily['high'] - daily['low']).rolling(14).mean()
    daily['atr_pct'] = daily['atr'] / daily['close'] * 100
    
    # Trend strength
    daily['slope_50'] = (daily['ma50'] - daily['ma50'].shift(10)) / daily['ma50'].shift(10) * 100
    daily['slope_200'] = (daily['ma200'] - daily['ma200'].shift(20)) / daily['ma200'].shift(20) * 100
    
    # Higher highs / Lower lows
    daily['hh_count'] = (daily['high'] > daily['high'].shift(1)).rolling(20).sum()
    daily['ll_count'] = (daily['low'] < daily['low'].shift(1)).rolling(20).sum()
    
    regime = pd.Series(index=daily.index, dtype=str)
    
    for i in range(200, len(daily)):
        price = daily['close'].iloc[i]
        ma50 = daily['ma50'].iloc[i]
        ma200 = daily['ma200'].iloc[i]
        slope_50 = daily['slope_50'].iloc[i]
        slope_200 = daily['slope_200'].iloc[i]
        atr_pct = daily['atr_pct'].iloc[i]
        hh = daily['hh_count'].iloc[i]
        ll = daily['ll_count'].iloc[i]
        
        # Strong Bull: Steep uptrend, high volatility, frequent HH
        if price > ma200 and slope_50 > 0.5 and slope_200 > 0.2 and hh > 12:
            regime.iloc[i] = "STRONG_BULL"
        
        # Weak Bull: Uptrend but shallow
        elif price > ma200 and ma50 > ma200 and slope_50 > 0:
            regime.iloc[i] = "WEAK_BULL"
        
        # Strong Bear: Steep downtrend, frequent LL
        elif price < ma200 and slope_50 < -0.5 and slope_200 < -0.2 and ll > 12:
            regime.iloc[i] = "STRONG_BEAR"
        
        # Weak Bear: Downtrend but shallow
        elif price < ma200 and ma50 < ma200 and slope_50 < 0:
            regime.iloc[i] = "WEAK_BEAR"
        
        # Consolidation: Low volatility, no clear trend
        elif atr_pct < 1.5:
            regime.iloc[i] = "CONSOLIDATION"
        
        # Default: Consolidation
        else:
            regime.iloc[i] = "CONSOLIDATION"
    
    return regime


# ── Map Regimes to 15M Data ────────────────────────────────────────────────────

def map_regime_to_15m(df_15m: pd.DataFrame, regime_daily: pd.Series) -> pd.Series:
    """Map daily regime labels to 15M bars."""
    regime_15m = pd.Series(index=df_15m.index, dtype=str)
    
    for date, regime in regime_daily.items():
        if pd.isna(regime):
            continue
        mask = df_15m.index.date == date.date()
        regime_15m[mask] = regime
    
    return regime_15m


# ── Run Backtest Per Regime ────────────────────────────────────────────────────

def run_regime_backtest(df_15m: pd.DataFrame, regime_label: str, params: BacktestParams):
    """Run backtest on a specific regime only."""
    regime_data = df_15m[df_15m['regime'] == regime_label].copy()
    
    if len(regime_data) < 1000:
        return None
    
    print(f"  Running backtest on {regime_label}...")
    print(f"    Data: {len(regime_data):,} candles ({regime_data.index[0].date()} → {regime_data.index[-1].date()})")
    
    bt = ScalpBacktest(regime_data, params=params)
    results = bt.run()
    
    return results


# ── Main Analysis ──────────────────────────────────────────────────────────────

def main():
    # Classify regimes
    print(f"  Classifying market regimes...\n")
    regime_daily = classify_regime_advanced(df_15m)
    regime_15m = map_regime_to_15m(df_15m, regime_daily)
    df_15m['regime'] = regime_15m
    
    # Count regime distribution
    regime_counts = regime_15m.value_counts()
    total = len(regime_15m)
    
    print(f"  Regime Distribution:")
    print(f"  {'─'*68}")
    for regime, count in regime_counts.items():
        if pd.isna(regime):
            continue
        pct = count / total * 100
        print(f"    {regime:<20} {count:>8,} candles ({pct:>5.1f}%)")
    print(f"  {'─'*68}\n")
    
    # Define backtest parameters
    params = BacktestParams(
        initial_balance=10_000,
        warmup_candles=300,
        step=1,
        min_signal_gap_candles=16,
        session_filter=True,
        min_soft_score=1,
        risk_params=ScalpRiskParams(
            risk_pct=0.0025,
            tp1_r=1.5,
            tp2_r=2.5,
            tp3_r=4.0,
        )
    )
    
    # Run backtest on each regime
    regime_results = {}
    
    for regime_label in ["STRONG_BULL", "WEAK_BULL", "CONSOLIDATION", "WEAK_BEAR", "STRONG_BEAR"]:
        if regime_label not in regime_counts.index:
            continue
        
        results = run_regime_backtest(df_15m, regime_label, params)
        if results:
            regime_results[regime_label] = results
    
    # Print comparison
    print(f"\n{'═'*70}")
    print(f"  REGIME PERFORMANCE COMPARISON")
    print(f"{'═'*70}")
    print(f"  {'Regime':<18} {'Trades':>7} {'WR%':>6} {'Exp R':>7} {'PF':>5} {'Return%':>8} {'MaxDD%':>7}")
    print(f"  {'─'*68}")
    
    for regime_label, results in regime_results.items():
        m = results['metrics']
        if m.get('total_trades', 0) == 0:
            print(f"  {regime_label:<18} {'No trades':>50}")
            continue
        
        print(f"  {regime_label:<18} {m['total_trades']:>7} {m['win_rate']:>6.1f} "
              f"{m['expectancy_r']:>+7.3f} {m['profit_factor']:>5.2f} "
              f"{m['total_return_pct']:>+8.2f} {m['max_drawdown_pct']:>7.2f}")
    
    print(f"{'═'*70}\n")
    
    # Recommendations
    print(f"  RECOMMENDATIONS:")
    print(f"  {'─'*68}")
    
    best_regime = None
    best_expectancy = -999
    worst_regime = None
    worst_expectancy = 999
    
    for regime_label, results in regime_results.items():
        m = results['metrics']
        if m.get('total_trades', 0) == 0:
            continue
        exp = m['expectancy_r']
        if exp > best_expectancy:
            best_expectancy = exp
            best_regime = regime_label
        if exp < worst_expectancy:
            worst_expectancy = exp
            worst_regime = regime_label
    
    if best_regime:
        print(f"  ✅  BEST regime:  {best_regime} ({best_expectancy:+.3f}R)")
        print(f"      → Trade aggressively in this regime")
    
    if worst_regime and worst_expectancy < 0:
        print(f"  🚨  WORST regime: {worst_regime} ({worst_expectancy:+.3f}R)")
        print(f"      → AVOID trading in this regime or reduce risk to 0.1%")
    
    print(f"  {'─'*68}\n")


if __name__ == "__main__":
    main()