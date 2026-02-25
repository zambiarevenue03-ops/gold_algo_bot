# main_signal_frequency.py

from utils.data_loader import load_csv
from backtest.signal_frequency import analyze_signal_frequency


def main():
    # ----------------------------------
    # Load processed 1H data
    # ----------------------------------
    df = load_csv("data/processed/XAUUSD_1H.csv")

    print("Data loaded.")
    print(f"Total candles: {len(df)}")
    print("Running signal frequency analysis...\n")

    # ----------------------------------
    # Analyze signal frequency
    # ----------------------------------
    signals_df = analyze_signal_frequency(df)

    # ----------------------------------
    # Handle results safely
    # ----------------------------------
    if signals_df.empty:
        print("No signals detected.")
        print("This indicates the strategy is currently over-filtered.")
        return

    # ----------------------------------
    # Basic inspection
    # ----------------------------------
    print("First few signals:")
    print(signals_df.head())

    print(f"\nTotal signals detected: {len(signals_df)}")

    # ----------------------------------
    # Signals per month
    # ----------------------------------
    signals_df["month"] = signals_df["timestamp"].dt.to_period("M")

    print("\nSignals per month:")
    print(signals_df.groupby("month").size())


if __name__ == "__main__":
    main()
