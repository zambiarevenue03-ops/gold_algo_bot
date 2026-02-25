# backtest/sanity_checks.py

def check_dataframe_integrity(df):
    assert not df.isnull().any().any(), "NaNs found in data"
    assert df.index.is_monotonic_increasing, "Index not increasing"
    assert len(df) > 100, "Not enough data"


def check_signal_frequency(signals, min_gap=5):
    """
    Ensure signals are not firing too frequently.
    """
    if len(signals) < 2:
        return True

    for i in range(1, len(signals)):
        if signals[i] - signals[i - 1] < min_gap:
            return False

    return True
