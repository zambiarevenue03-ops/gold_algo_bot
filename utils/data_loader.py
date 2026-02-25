# utils/data_loader.py
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    # ----------------------------------
    # Load raw MT5 file (single column)
    # ----------------------------------
    raw = pd.read_csv(path, header=0)

    # Split the single column by tabs
    df = raw.iloc[:, 0].str.split("\t", expand=True)

    df.columns = [
        "date", "time", "open", "high", "low",
        "close", "tickvol", "vol", "spread"
    ]

    # ----------------------------------
    # Build datetime index
    # ----------------------------------
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y.%m.%d %H:%M:%S"
    )

    df.set_index("datetime", inplace=True)

    # ----------------------------------
    # Convert numeric columns
    # ----------------------------------
    numeric_cols = ["open", "high", "low", "close", "tickvol"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # ----------------------------------
    # Standardize OHLCV
    # ----------------------------------
    df = df.rename(columns={"tickvol": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]

    # ----------------------------------
    # Safety checks
    # ----------------------------------
    assert len(df) > 1000, "Dataframe too small — check CSV contents"
    assert df.index.is_monotonic_increasing, "Datetime index not sorted"

    return df
