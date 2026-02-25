"""
backtest/parameter_stability.py
=================================
Parameter stability testing via grid search.

Tests if strategy edge holds across slight parameter variations.

Usage:
    from backtest.parameter_stability import ParameterGrid, run_stability_test
    
    grid = ParameterGrid(
        tp1_r=[1.3, 1.5, 1.7],
        tp2_r=[2.3, 2.5, 2.7],
        sl_buffer_atr=[0.2, 0.3, 0.4],
    )
    
    results = run_stability_test(df_15m, grid, base_params)
    results.print_summary()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from itertools import product
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.scalping_backtest import ScalpBacktest, BacktestParams
from scalping.risk_model import ScalpRiskParams


@dataclass
class GridResult:
    """Single backtest result from parameter grid."""
    params:         Dict[str, Any]
    total_return:   float
    expectancy_r:   float
    win_rate:       float
    max_dd:         float
    total_trades:   int
    profit_factor:  float


class ParameterGrid:
    """Defines parameter ranges to test."""
    
    def __init__(self, **param_ranges):
        self.param_ranges = param_ranges
        self.param_names = list(param_ranges.keys())
        self.combinations = list(product(*param_ranges.values()))
    
    def __len__(self):
        return len(self.combinations)
    
    def __iter__(self):
        for combo in self.combinations:
            yield dict(zip(self.param_names, combo))


class StabilityTestResults:
    """Container for parameter stability test results."""
    
    def __init__(self, results: List[GridResult], base_params: Dict[str, Any]):
        self.results = results
        self.base_params = base_params
        
        self.results_sorted = sorted(results, key=lambda x: x.total_return, reverse=True)
        self.best = self.results_sorted[0]
        self.worst = self.results_sorted[-1]
        
        returns = [r.total_return for r in results]
        self.mean_return = np.mean(returns)
        self.median_return = np.median(returns)
        self.std_return = np.std(returns)
        
        self.profitable_count = sum(1 for r in results if r.total_return > 0)
        self.profitable_pct = self.profitable_count / len(results) * 100
        
        if self.profitable_pct >= 80:
            self.robustness = "EXCELLENT"
        elif self.profitable_pct >= 60:
            self.robustness = "GOOD"
        elif self.profitable_pct >= 40:
            self.robustness = "FAIR"
        else:
            self.robustness = "POOR"
    
    def print_summary(self):
        """Print formatted stability test summary."""
        print(f"\n{'═'*70}")
        print(f"  PARAMETER STABILITY TEST RESULTS")
        print(f"{'═'*70}")
        print(f"  Total combinations tested:  {len(self.results)}")
        print(f"  Profitable combinations:    {self.profitable_count}/{len(self.results)} ({self.profitable_pct:.1f}%)")
        print(f"  Robustness rating:          {self.robustness}")
        print(f"\n{'─'*70}")
        print(f"  RETURN DISTRIBUTION")
        print(f"{'─'*70}")
        print(f"  Mean return:     {self.mean_return:+.2f}%")
        print(f"  Median return:   {self.median_return:+.2f}%")
        print(f"  Std deviation:   {self.std_return:.2f}%")
        print(f"  Best:            {self.best.total_return:+.2f}%")
        print(f"  Worst:           {self.worst.total_return:+.2f}%")
        
        print(f"\n{'─'*70}")
        print(f"  BEST PARAMETER SET")
        print(f"{'─'*70}")
        for k, v in self.best.params.items():
            print(f"    {k:<20} {v}")
        print(f"    {'return':<20} {self.best.total_return:+.2f}%")
        print(f"    {'expectancy':<20} {self.best.expectancy_r:+.3f}R")
        print(f"    {'win_rate':<20} {self.best.win_rate:.1f}%")
        
        print(f"\n{'─'*70}")
        print(f"  INTERPRETATION")
        print(f"{'─'*70}")
        
        if self.profitable_pct >= 80:
            print(f"  ✅  Strategy is ROBUST. {self.profitable_pct:.0f}% of parameter")
            print(f"      combinations are profitable. Edge is real, not curve-fitted.")
        elif self.profitable_pct >= 60:
            print(f"  ✅  Strategy is MODERATELY ROBUST. Most combinations profitable.")
        elif self.profitable_pct >= 40:
            print(f"  ⚠️  Strategy is SOMEWHAT FRAGILE. Only {self.profitable_pct:.0f}% work.")
        else:
            print(f"  🚨  Strategy is HIGHLY FRAGILE. Only {self.profitable_pct:.0f}% profitable.")
        
        print(f"\n{'═'*70}\n")
    
    def print_top_10(self):
        """Print top 10 parameter combinations."""
        print(f"\n  TOP 10 PARAMETER COMBINATIONS")
        print(f"  {'─'*68}")
        print(f"  {'Rank':<6} {'Return':>8} {'Exp R':>8} {'WR%':>6} {'Params':<40}")
        print(f"  {'─'*68}")
        
        for i, r in enumerate(self.results_sorted[:10], 1):
            params_str = ', '.join([f"{k}={v}" for k, v in r.params.items()])
            if len(params_str) > 40:
                params_str = params_str[:37] + "..."
            print(f"  {i:<6} {r.total_return:>+7.2f}% {r.expectancy_r:>+7.3f}R {r.win_rate:>5.1f}% {params_str}")
        
        print(f"  {'─'*68}\n")


def run_stability_test(
    df_15m: pd.DataFrame,
    param_grid: ParameterGrid,
    base_params: BacktestParams,
) -> StabilityTestResults:
    """Run parameter stability test."""
    results = []
    total = len(param_grid)
    
    print(f"\n{'═'*70}")
    print(f"  PARAMETER STABILITY TEST")
    print(f"{'═'*70}")
    print(f"  Testing {total} parameter combinations...")
    print(f"  Estimated time: ~{total * 15} seconds\n")
    
    for i, params in enumerate(param_grid, 1):
        print(f"  Progress: {i}/{total}", end='\r')
        
        test_params = BacktestParams(
            initial_balance    = base_params.initial_balance,
            warmup_candles     = base_params.warmup_candles,
            step               = base_params.step,
            min_signal_gap_candles = base_params.min_signal_gap_candles,
            session_filter     = base_params.session_filter,
            utc_offset_hours   = base_params.utc_offset_hours,
            min_soft_score     = base_params.min_soft_score,
            pivot_lookback     = params.get('pivot_lookback', base_params.pivot_lookback),
            obv_ma_period      = params.get('obv_ma_period', base_params.obv_ma_period),
            atr_period         = params.get('atr_period', base_params.atr_period),
            atr_ma_period      = params.get('atr_ma_period', base_params.atr_ma_period),
            risk_params        = ScalpRiskParams(
                risk_pct       = params.get('risk_pct', base_params.risk_params.risk_pct),
                sl_buffer_atr  = params.get('sl_buffer_atr', base_params.risk_params.sl_buffer_atr),
                tp1_r          = params.get('tp1_r', base_params.risk_params.tp1_r),
                tp2_r          = params.get('tp2_r', base_params.risk_params.tp2_r),
                tp3_r          = params.get('tp3_r', base_params.risk_params.tp3_r),
                sl_max_distance = base_params.risk_params.sl_max_distance,
            ),
        )
        
        bt = ScalpBacktest(df_15m, params=test_params)
        result = bt.run()
        m = result['metrics']
        
        results.append(GridResult(
            params         = params.copy(),
            total_return   = m.get('total_return_pct', 0.0),
            expectancy_r   = m.get('expectancy_r', 0.0),
            win_rate       = m.get('win_rate', 0.0),
            max_dd         = m.get('max_drawdown_pct', 0.0),
            total_trades   = m.get('total_trades', 0),
            profit_factor  = m.get('profit_factor', 0.0),
        ))
    
    print(f"\n  Progress: {total}/{total} ✓\n")
    
    return StabilityTestResults(results, base_params.__dict__)