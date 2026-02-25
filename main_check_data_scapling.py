import os
import sys
import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

TIMEFRAMES = {
    "4H":  {"file": "XAUUSD_4H.csv",  "minutes": 240, "min_candles": 3000,  "tolerance_pct": 0.30},
    "1H":  {"file": "XAUUSD_1H.csv",  "minutes": 60,  "min_candles": 10000, "tolerance_pct": 0.30},
    "15M": {"file": "XAUUSD_15M.csv", "minutes": 15,  "min_candles": 30000, "tolerance_pct": 0.30},
    "5M":  {"file": "XAUUSD_5M.csv",  "minutes": 5,   "min_candles": 80000, "tolerance_pct": 0.30},
}

GOLD_PRICE_MIN = 1000.0
GOLD_PRICE_MAX = 4000.0

# ── colours ────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✗{RESET}  {RED}{msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET}  {YELLOW}{msg}{RESET}")
def info(msg):  print(f"  {CYAN}→{RESET}  {msg}")

# ── MT5 CSV parser (same logic as existing data_loader.py) ─────────────────────
def load_mt5_csv(path: str) -> pd.DataFrame:
    """
    Handles MT5's tab-delimited format:
      <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    Returns a clean DataFrame indexed by datetime with columns:
      open, high, low, close, volume
    """
    raw = pd.read_csv(path, header=0)

    # Detect tab-delimited (single column) vs already parsed
    if raw.shape[1] == 1:
        df = raw.iloc[:, 0].str.split("\t", expand=True)
        df.columns = ["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"]
    elif raw.shape[1] >= 6:
        # Already comma-separated — try to adapt column names
        df = raw.copy()
        df.columns = [c.strip().lower().replace("<", "").replace(">", "") for c in df.columns]
        if "date" not in df.columns and df.columns[0] not in ("date",):
            # First column might be combined datetime
            df.insert(1, "time", "00:00:00")
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        if "tickvol" not in df.columns:
            df["tickvol"] = 0
    else:
        raise ValueError(f"Unexpected column count: {raw.shape[1]}")

    # Build datetime index
    if "time" in df.columns and df["time"].str.len().max() > 4:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format="%Y.%m.%d %H:%M:%S",
            errors="coerce"
        )
    else:
        df["datetime"] = pd.to_datetime(df["date"].astype(str), errors="coerce")

    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime")
    df = df.sort_index()

    # Coerce numeric columns
    for col in ["open", "high", "low", "close", "tickvol"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={"tickvol": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]

    return df


# ── individual checks ──────────────────────────────────────────────────────────
def check_file_exists(path):
    if os.path.isfile(path):
        size_kb = os.path.getsize(path) / 1024
        ok(f"File found ({size_kb:.0f} KB)")
        return True
    else:
        fail(f"File NOT found: {path}")
        return False


def check_loads(path):
    try:
        df = load_mt5_csv(path)
        ok(f"Loaded successfully — {len(df):,} rows")
        return df
    except Exception as e:
        fail(f"Failed to load: {e}")
        return None


def check_columns(df):
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if not missing:
        ok(f"All required columns present: {list(required)}")
        return True
    else:
        fail(f"Missing columns: {missing}")
        return False


def check_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        fail("Index is not a DatetimeIndex")
        return False
    if df.index.is_monotonic_increasing:
        ok("Index is sorted ascending")
    else:
        fail("Index is NOT sorted — timestamps out of order")
        return False
    nulls = df.index.isnull().sum()
    if nulls == 0:
        ok("No null timestamps")
    else:
        fail(f"{nulls:,} null timestamps in index")
        return False
    return True


def check_nan(df):
    total_nan = df[["open", "high", "low", "close"]].isnull().sum().sum()
    if total_nan == 0:
        ok("No NaN values in OHLC columns")
    else:
        warn(f"{total_nan:,} NaN values found — may cause calculation errors")
    return total_nan == 0


def check_candle_count(df, tf_name, min_candles):
    n = len(df)
    if n >= min_candles:
        ok(f"Candle count: {n:,} (minimum {min_candles:,}) ✓")
        return True
    else:
        warn(f"Candle count: {n:,} — below recommended minimum of {min_candles:,}")
        return False


def check_ohlc_sanity(df):
    errors = 0
    # high must be >= low
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        fail(f"{bad_hl:,} candles where high < low")
        errors += bad_hl
    # high must be >= open and close
    bad_ho = (df["high"] < df["open"]).sum()
    bad_hc = (df["high"] < df["close"]).sum()
    if bad_ho + bad_hc > 0:
        fail(f"{bad_ho + bad_hc:,} candles where high < open or close")
        errors += bad_ho + bad_hc
    # low must be <= open and close
    bad_lo = (df["low"] > df["open"]).sum()
    bad_lc = (df["low"] > df["close"]).sum()
    if bad_lo + bad_lc > 0:
        fail(f"{bad_lo + bad_lc:,} candles where low > open or close")
        errors += bad_lo + bad_lc
    if errors == 0:
        ok("OHLC sanity: all candles valid (high >= low, etc.)")
    return errors == 0


def check_price_range(df):
    price_min = df["close"].min()
    price_max = df["close"].max()
    info(f"Price range: {price_min:.2f} – {price_max:.2f}")
    if price_min >= GOLD_PRICE_MIN and price_max <= GOLD_PRICE_MAX:
        ok(f"Price range looks correct for XAUUSD")
    else:
        warn(f"Unusual price range — expected {GOLD_PRICE_MIN}–{GOLD_PRICE_MAX} for gold")


def check_spacing(df, tf_name, expected_minutes, tolerance_pct):
    diffs = df.index.to_series().diff().dropna()
    median_diff = diffs.median()
    expected = pd.Timedelta(minutes=expected_minutes)
    ratio = abs(median_diff - expected) / expected

    info(f"Median candle interval: {median_diff}")
    if ratio <= tolerance_pct:
        ok(f"Candle spacing matches {tf_name} ({expected_minutes}min expected)")
    else:
        fail(f"Candle spacing mismatch — expected ~{expected}, got {median_diff}")

    # Count gaps larger than 3× expected (market closed gaps are fine, bigger ones are data holes)
    big_gaps = (diffs > expected * 6).sum()
    if big_gaps > 0:
        warn(f"{big_gaps:,} large gaps detected (> 6× expected interval) — check for missing data")
    else:
        ok("No abnormal data gaps detected")


def check_date_coverage(loaded_dfs):
    print(f"\n{BOLD}{CYAN}── Temporal Alignment Across All Timeframes ──{RESET}")
    starts = {}
    ends = {}
    for tf_name, df in loaded_dfs.items():
        starts[tf_name] = df.index[0]
        ends[tf_name] = df.index[-1]
        info(f"{tf_name:>4s}: {df.index[0].date()}  →  {df.index[-1].date()}  ({len(df):>7,} candles)")

    all_starts = list(starts.values())
    all_ends = list(ends.values())
    latest_start = max(all_starts)
    earliest_end = min(all_ends)

    if latest_start < earliest_end:
        overlap_days = (earliest_end - latest_start).days
        ok(f"Common date range: {latest_start.date()} → {earliest_end.date()} ({overlap_days:,} days overlap)")
    else:
        fail("Timeframes do NOT share a common date range — backtesting will fail")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  XAUUSD SCALPING BOT — DATA VALIDATION{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")
    print(f"Data directory: {DATA_DIR}\n")

    loaded_dfs = {}
    all_passed = True

    for tf_name, cfg in TIMEFRAMES.items():
        path = os.path.join(DATA_DIR, cfg["file"])
        print(f"{BOLD}{'─'*60}{RESET}")
        print(f"{BOLD}  {tf_name}  →  {cfg['file']}{RESET}")
        print(f"{'─'*60}")

        # 1. File exists
        if not check_file_exists(path):
            all_passed = False
            print()
            continue

        # 2. Loads without error
        df = check_loads(path)
        if df is None:
            all_passed = False
            print()
            continue

        # 3. Columns
        if not check_columns(df):
            all_passed = False

        # 4. Index
        if not check_index(df):
            all_passed = False

        # 5. NaN
        check_nan(df)

        # 6. Candle count
        check_candle_count(df, tf_name, cfg["min_candles"])

        # 7. OHLC sanity
        if not check_ohlc_sanity(df):
            all_passed = False

        # 8. Price range
        check_price_range(df)

        # 9. Spacing
        check_spacing(df, tf_name, cfg["minutes"], cfg["tolerance_pct"])

        loaded_dfs[tf_name] = df
        print()

    # 10. Cross-timeframe alignment
    if len(loaded_dfs) >= 2:
        check_date_coverage(loaded_dfs)

    # ── final verdict ──────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    if all_passed and len(loaded_dfs) == len(TIMEFRAMES):
        print(f"{BOLD}{GREEN}  ALL CHECKS PASSED — Data is ready for scalping bot build{RESET}")
    elif len(loaded_dfs) < len(TIMEFRAMES):
        missing = set(TIMEFRAMES.keys()) - set(loaded_dfs.keys())
        print(f"{BOLD}{RED}  MISSING TIMEFRAMES: {missing}{RESET}")
        print(f"{RED}  Place MT5 exports in:  {DATA_DIR}{RESET}")
        print(f"\n  Expected filenames:")
        for tf_name, cfg in TIMEFRAMES.items():
            print(f"    {cfg['file']}")
    else:
        print(f"{BOLD}{YELLOW}  CHECKS COMPLETED WITH WARNINGS — Review output above{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()