"""
run_monte_carlo.py
===================
Run Monte Carlo simulation on your backtest results.

Usage:
    python run_monte_carlo.py
"""

import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pandas as pd
from backtest.monte_carlo import MonteCarloSimulator

# ── Configuration ──────────────────────────────────────────────────────────────

# Path to your trade journal (from backtest)
JOURNAL_PATH = os.path.join(ROOT, "backtest", "results", "scalp_journal_full_history.csv")

# Initial balance used in backtest
INITIAL_BALANCE = 10_000.0

# Number of simulations (2000 = good balance of speed vs accuracy)
N_SIMULATIONS = 2000

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*70}")
    print(f"  MONTE CARLO SIMULATION")
    print(f"{'═'*70}")
    print(f"  Journal:         {JOURNAL_PATH}")
    print(f"  Initial balance: ${INITIAL_BALANCE:,.2f}")
    print(f"  Simulations:     {N_SIMULATIONS:,}")
    print(f"{'═'*70}\n")
    
    # Check file exists
    if not os.path.exists(JOURNAL_PATH):
        print(f"  ERROR: Trade journal not found at:")
        print(f"  {JOURNAL_PATH}")
        print(f"\n  Run main_scalp_backtest.py first to generate the journal.")
        return
    
    # Load trades
    df = pd.read_csv(JOURNAL_PATH)
    trades = df.to_dict('records')
    
    print(f"  Loaded {len(trades)} trades from journal\n")
    
    # Run Monte Carlo
    mc = MonteCarloSimulator(trades, initial_balance=INITIAL_BALANCE)
    results = mc.run(n_simulations=N_SIMULATIONS, seed=42)
    
    # Print results
    mc.print_summary(results)
    mc.print_histogram(results, bins=20)
    
    # Additional analysis for $500 account
    print(f"\n{'═'*70}")
    print(f"  ACCOUNT SIZE ANALYSIS")
    print(f"{'═'*70}\n")
    
    # Test at different account sizes
    for balance in [500, 1000, 2000, 5000, 10000]:
        # Scale risk proportionally
        scaled_trades = []
        for t in trades:
            scaled_t = t.copy()
            scaled_t['risk_usd'] = t['risk_usd'] * (balance / INITIAL_BALANCE)
            scaled_t['net_pnl'] = t['net_pnl'] * (balance / INITIAL_BALANCE)
            scaled_trades.append(scaled_t)
        
        mc_scaled = MonteCarloSimulator(scaled_trades, initial_balance=balance)
        results_scaled = mc_scaled.run(n_simulations=1000, seed=42)
        
        print(f"  ${balance:>6,} account:")
        print(f"    Median return:    {results_scaled.median_return:>+7.2f}%")
        print(f"    Worst 5%:         {results_scaled.worst_5pct:>+7.2f}%")
        print(f"    Median max DD:    {results_scaled.median_dd:>7.2f}%")
        print(f"    Worst 5% DD:      {results_scaled.worst_dd_5pct:>7.2f}%")
        print(f"    Risk of ruin:     {results_scaled.ruin_probability:>7.2f}%")
        print()
    
    print(f"{'═'*70}\n")
    
    # Recommendation
    print(f"  RECOMMENDATION:")
    print(f"  ──────────────────────────────────────────────────────────")
    
    if results.ruin_probability > 1.0:
        print(f"  ⚠️  At $10k: {results.ruin_probability:.1f}% risk of ruin detected.")
        print(f"      Consider reducing risk per trade or increasing capital.")
    else:
        print(f"  ✅  At $10k: Risk of ruin is negligible ({results.ruin_probability:.2f}%).")
    
    print()
    
    # Check $500 viability
    mc_500 = MonteCarloSimulator(
        [{**t, 'risk_usd': t['risk_usd'] * 0.05, 'net_pnl': t['net_pnl'] * 0.05} for t in trades],
        initial_balance=500
    )
    results_500 = mc_500.run(n_simulations=1000, seed=42)
    
    if results_500.ruin_probability > 5.0:
        print(f"  🚨  At $500: {results_500.ruin_probability:.1f}% risk of ruin.")
        print(f"      This account size is TOO SMALL for 0.25% risk.")
        print(f"      Minimum recommended: $2,000")
    elif results_500.ruin_probability > 1.0:
        print(f"  ⚠️  At $500: {results_500.ruin_probability:.1f}% risk of ruin.")
        print(f"      Viable but risky. Reduce to 0.10% risk or save to $2,000.")
    else:
        print(f"  ✅  At $500: {results_500.ruin_probability:.2f}% risk of ruin.")
        print(f"      Account size is viable with current risk settings.")
    
    print(f"  ──────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()