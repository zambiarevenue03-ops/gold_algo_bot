"""
backtest/monte_carlo.py
========================
Monte Carlo simulation for trade sequence randomization.

What it does:
  1. Takes actual backtest trades (W/L/R-multiples)
  2. Shuffles them randomly 1,000+ times
  3. Recalculates equity curve for each shuffle
  4. Shows distribution of outcomes (best/worst/median)

Why it matters:
  Your backtest shows +13.8% over 3 years.
  But what if all wins happened first and losses last?
  Monte Carlo reveals if you got lucky with trade sequence.

Key metrics:
  - Median return (50th percentile)
  - Worst-case return (5th percentile)
  - Max drawdown distribution
  - Probability of ruin
  - Confidence intervals

Output:
  - Summary statistics
  - Histogram of returns
  - Drawdown distribution
  - Risk of ruin at different account sizes

Usage:
    from backtest.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator(trades, initial_balance=10_000)
    results = mc.run(n_simulations=1000)
    mc.print_summary(results)
    mc.print_histogram(results)
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class MonteCarloResults:
    """Container for Monte Carlo simulation results."""
    n_simulations:      int
    initial_balance:    float
    
    # Return distribution
    returns:            np.ndarray      # array of final returns (%)
    median_return:      float           # 50th percentile
    mean_return:        float
    worst_5pct:         float           # 5th percentile (worst realistic case)
    best_5pct:          float           # 95th percentile
    
    # Drawdown distribution
    max_drawdowns:      np.ndarray      # array of max DD (%) per simulation
    median_dd:          float
    worst_dd_5pct:      float           # 95th percentile DD
    
    # Risk of ruin
    ruin_count:         int             # simulations that hit 0
    ruin_probability:   float           # % that went bust
    
    # Original vs simulated
    original_return:    float
    original_max_dd:    float
    
    # Confidence assessment
    better_than_original: int           # count of sims better than actual
    confidence_level:   float           # % of sims >= original


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trade sequence randomization.
    
    Takes actual trades and reshuffles them to test if results
    were due to lucky sequencing or genuine edge.
    """
    
    def __init__(self, trades: List[Dict], initial_balance: float = 10_000.0):
        """
        Parameters
        ----------
        trades : list of dicts
            Each dict must have: 'net_pnl', 'risk_usd', 'r_multiple'
        initial_balance : float
            Starting account size
        """
        self.trades = trades
        self.initial_balance = initial_balance
        
        # Extract trade outcomes as R-multiples for easy randomization
        self.r_multiples = np.array([t['r_multiple'] for t in trades])
        self.risk_amounts = np.array([t['risk_usd'] for t in trades])
        
        # Original equity curve for comparison
        self.original_curve = self._simulate_equity_curve(
            self.r_multiples, self.risk_amounts
        )
        self.original_return = ((self.original_curve[-1] - initial_balance) 
                                / initial_balance * 100)
        self.original_max_dd = self._calculate_max_drawdown(self.original_curve)
    
    def run(self, n_simulations: int = 1000, seed: int = None) -> MonteCarloResults:
        """
        Run Monte Carlo simulation.
        
        Parameters
        ----------
        n_simulations : int
            Number of random shuffles to test (1000-10000)
        seed : int
            Random seed for reproducibility
        
        Returns
        -------
        MonteCarloResults
        """
        rng = np.random.default_rng(seed)
        
        returns = []
        max_drawdowns = []
        ruin_count = 0
        better_count = 0
        
        print(f"\n  Running Monte Carlo: {n_simulations:,} simulations...")
        
        for i in range(n_simulations):
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1:,}/{n_simulations:,}", end='\r')
            
            # Shuffle trade order
            indices = rng.permutation(len(self.r_multiples))
            shuffled_r = self.r_multiples[indices]
            shuffled_risk = self.risk_amounts[indices]
            
            # Simulate equity curve with shuffled trades
            equity_curve = self._simulate_equity_curve(shuffled_r, shuffled_risk)
            
            # Check for ruin
            if equity_curve.min() <= 0:
                ruin_count += 1
                final_return = -100.0
            else:
                final_return = ((equity_curve[-1] - self.initial_balance) 
                               / self.initial_balance * 100)
            
            returns.append(final_return)
            max_drawdowns.append(self._calculate_max_drawdown(equity_curve))
            
            if final_return >= self.original_return:
                better_count += 1
        
        print(f"    Progress: {n_simulations:,}/{n_simulations:,} ✓\n")
        
        # Convert to numpy arrays for stats
        returns = np.array(returns)
        max_drawdowns = np.array(max_drawdowns)
        
        return MonteCarloResults(
            n_simulations      = n_simulations,
            initial_balance    = self.initial_balance,
            returns            = returns,
            median_return      = float(np.median(returns)),
            mean_return        = float(np.mean(returns)),
            worst_5pct         = float(np.percentile(returns, 5)),
            best_5pct          = float(np.percentile(returns, 95)),
            max_drawdowns      = max_drawdowns,
            median_dd          = float(np.median(max_drawdowns)),
            worst_dd_5pct      = float(np.percentile(max_drawdowns, 95)),
            ruin_count         = ruin_count,
            ruin_probability   = ruin_count / n_simulations * 100,
            original_return    = self.original_return,
            original_max_dd    = self.original_max_dd,
            better_than_original = better_count,
            confidence_level   = better_count / n_simulations * 100,
        )
    
    def _simulate_equity_curve(self, r_multiples: np.ndarray, 
                                risk_amounts: np.ndarray) -> np.ndarray:
        """
        Simulate equity curve from sequence of R-multiples.
        
        Uses actual risk amounts (not fixed %) to account for
        position sizing changes as balance grows/shrinks.
        """
        balance = self.initial_balance
        curve = [balance]
        
        for r, risk in zip(r_multiples, risk_amounts):
            pnl = r * risk
            balance += pnl
            curve.append(balance)
        
        return np.array(curve)
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate max drawdown % from equity curve."""
        peak = equity_curve[0]
        max_dd = 0.0
        
        for balance in equity_curve:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def print_summary(self, results: MonteCarloResults) -> None:
        """Print formatted Monte Carlo results."""
        r = results
        
        print(f"\n{'═'*60}")
        print(f"  MONTE CARLO SIMULATION RESULTS")
        print(f"{'═'*60}")
        print(f"  Simulations:      {r.n_simulations:,}")
        print(f"  Initial balance:  ${r.initial_balance:,.2f}")
        print(f"  Total trades:     {len(self.trades)}")
        print(f"\n{'─'*60}")
        print(f"  RETURN DISTRIBUTION")
        print(f"{'─'*60}")
        print(f"  Original return:  {r.original_return:+.2f}%")
        print(f"  Mean (MC):        {r.mean_return:+.2f}%")
        print(f"  Median (MC):      {r.median_return:+.2f}%")
        print(f"  Best 5%:          {r.best_5pct:+.2f}%")
        print(f"  Worst 5%:         {r.worst_5pct:+.2f}%")
        print(f"\n{'─'*60}")
        print(f"  DRAWDOWN DISTRIBUTION")
        print(f"{'─'*60}")
        print(f"  Original max DD:  {r.original_max_dd:.2f}%")
        print(f"  Median DD (MC):   {r.median_dd:.2f}%")
        print(f"  Worst 5% DD:      {r.worst_dd_5pct:.2f}%")
        print(f"\n{'─'*60}")
        print(f"  RISK ASSESSMENT")
        print(f"{'─'*60}")
        print(f"  Ruin events:      {r.ruin_count}/{r.n_simulations} ({r.ruin_probability:.2f}%)")
        print(f"  Better than original: {r.better_than_original}/{r.n_simulations} ({r.confidence_level:.1f}%)")
        
        # Interpretation
        print(f"\n{'─'*60}")
        print(f"  INTERPRETATION")
        print(f"{'─'*60}")
        
        if r.confidence_level > 60:
            print(f"  ⚠️  Your actual result ({r.original_return:+.2f}%) was ABOVE the")
            print(f"      median ({r.median_return:+.2f}%). You may have gotten lucky")
            print(f"      with trade sequencing.")
        elif r.confidence_level < 40:
            print(f"  ⚠️  Your actual result ({r.original_return:+.2f}%) was BELOW the")
            print(f"      median ({r.median_return:+.2f}%). Random sequencing could")
            print(f"      have produced better results.")
        else:
            print(f"  ✅  Your actual result ({r.original_return:+.2f}%) is near the")
            print(f"      median ({r.median_return:+.2f}%). Results are typical,")
            print(f"      not due to lucky sequencing.")
        
        print()
        
        if r.ruin_probability > 1.0:
            print(f"  ⚠️  {r.ruin_probability:.1f}% of simulations went to zero.")
            print(f"      Consider reducing risk per trade or increasing capital.")
        elif r.ruin_probability > 0.1:
            print(f"  ⚠️  {r.ruin_probability:.2f}% risk of ruin detected.")
            print(f"      Account size may be insufficient for this strategy.")
        else:
            print(f"  ✅  Risk of ruin is negligible ({r.ruin_probability:.2f}%).")
        
        print()
        
        if abs(r.median_return - r.original_return) > 3.0:
            print(f"  ⚠️  Large gap between median ({r.median_return:+.2f}%) and")
            print(f"      actual ({r.original_return:+.2f}%). Results may be luck-dependent.")
        else:
            print(f"  ✅  Median return close to actual. Strategy is robust to")
            print(f"      trade sequencing.")
        
        print(f"\n{'─'*60}")
        print(f"  REALISTIC EXPECTATIONS")
        print(f"{'─'*60}")
        print(f"  If you ran this strategy 100 times:")
        print(f"    • 50 times you'd get:  {r.median_return:+.2f}% (median)")
        print(f"    • 5 times you'd get:   {r.worst_5pct:+.2f}% or worse")
        print(f"    • 5 times you'd get:   {r.best_5pct:+.2f}% or better")
        print(f"    • Max DD would likely be: {r.median_dd:.2f}% to {r.worst_dd_5pct:.2f}%")
        print(f"{'═'*60}\n")
    
    def print_histogram(self, results: MonteCarloResults, bins: int = 20) -> None:
        """Print ASCII histogram of return distribution."""
        returns = results.returns
        
        print(f"\n  RETURN DISTRIBUTION HISTOGRAM")
        print(f"  {'─'*58}")
        
        hist, edges = np.histogram(returns, bins=bins)
        max_count = hist.max()
        
        for i in range(len(hist)):
            bin_start = edges[i]
            bin_end = edges[i + 1]
            count = hist[i]
            bar_len = int(count / max_count * 40)
            bar = '█' * bar_len
            
            # Mark original return
            marker = ''
            if bin_start <= results.original_return <= bin_end:
                marker = ' ← YOUR RESULT'
            
            print(f"  {bin_start:+6.1f}% to {bin_end:+6.1f}%  {bar:40s} {count:4d}{marker}")
        
        print(f"  {'─'*58}\n")


def run_monte_carlo_from_journal(journal_path: str, initial_balance: float = 10_000,
                                  n_simulations: int = 1000) -> MonteCarloResults:
    """
    Convenience function to run Monte Carlo directly from CSV journal.
    
    Parameters
    ----------
    journal_path : str
        Path to trade journal CSV (from backtest)
    initial_balance : float
        Starting account size
    n_simulations : int
        Number of Monte Carlo runs
    
    Returns
    -------
    MonteCarloResults
    """
    df = pd.read_csv(journal_path)
    trades = df.to_dict('records')
    
    mc = MonteCarloSimulator(trades, initial_balance=initial_balance)
    results = mc.run(n_simulations=n_simulations)
    mc.print_summary(results)
    mc.print_histogram(results)
    
    return results