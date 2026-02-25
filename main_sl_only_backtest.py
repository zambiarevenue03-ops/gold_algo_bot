# main_sl_only_backtest.py

from utils.data_loader import load_csv
from backtest.sl_only_backtest import run_sl_only_backtest


def main():
    df = load_csv("data/processed/XAUUSD_1H.csv")

    print("Running SL-only backtest...")
    trades = run_sl_only_backtest(df)

    if trades.empty:
        print("No trades found.")
        return

    print("\nBacktest Summary:")
    print(trades.head())

    total = len(trades)
    sl_hits = trades["sl_hit"].sum()
    survival = total - sl_hits

    print(f"\nTotal trades: {total}")
    print(f"SL hit trades: {sl_hits}")
    print(f"Survived trades: {survival}")
    print(f"Survival rate: {survival / total:.2%}")

    print("\nAverage risk per trade:")
    print(trades["risk"].describe())


if __name__ == "__main__":
    main()
