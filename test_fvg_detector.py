import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scalping.fvg_detector import (
    find_fvgs,
    get_latest_fvg,
    price_in_fvg,
    fvg_has_been_mitigated,
    price_approaching_fvg,
    summarise_fvgs,
)

# -- colour helpers -------------------------------------------------------------
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
BOLD = "\033[1m"; RESET = "\033[0m"

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  {GREEN}OK{RESET}  {name}")
        passed += 1
    else:
        print(f"  {RED}X{RESET}  {RED}{name}{RESET}" + (f" - {detail}" if detail else ""))
        failed += 1


# -- synthetic data builders ----------------------------------------------------

def make_df(rows: list) -> pd.DataFrame:
    """
    Build a DataFrame from a list of (open, high, low, close) tuples.
    Volume is set to 1000 throughout.
    Timestamps are hourly starting from 2025-01-01.
    """
    dates = pd.date_range("2025-01-01", periods=len(rows), freq="1h")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=dates)
    df["volume"] = 1000
    return df


def flat_candles(n, base=2500.0, spread=5.0):
    """Generate n quiet candles around a base price."""
    return [(base, base + spread, base - spread, base)] * n


# -- TEST 1: Bullish FVG detection ----------------------------------------------
print(f"\n{BOLD}-- TEST 1: Bullish FVG Detection --{RESET}")

# Pattern:
#   candle[-3]: high = 2510  (candle before gap)
#   candle[-2]: huge bullish impulse  (creates the FVG)
#   candle[-1]: low = 2530  (candle after gap) → gap = 2510 to 2530
rows = flat_candles(30) + [
    (2500, 2510, 2498, 2508),   # i-2: high = 2510
    (2510, 2560, 2509, 2555),   # i-1: impulse candle
    (2555, 2565, 2530, 2562),   # i:   low = 2530  → FVG zone: 2510-2530
]
df_bull = make_df(rows)
fvgs = find_fvgs(df_bull, "bullish", lookback=10)

check("Bullish FVG found", len(fvgs) >= 1, f"found {len(fvgs)}")
if fvgs:
    f = fvgs[0]
    check("FVG type is bullish",       f["type"] == "bullish",  f"got {f['type']}")
    check("FVG low = 2510 (prev2 high)", abs(f["fvg_low"] - 2510) < 0.01, f"got {f['fvg_low']}")
    check("FVG high = 2530 (curr low)",  abs(f["fvg_high"] - 2530) < 0.01, f"got {f['fvg_high']}")
    check("Gap size = 20",              abs(f["gap_size"] - 20) < 0.01,   f"got {f['gap_size']}")
    check("Midpoint = 2520",            abs(f["midpoint"] - 2520) < 0.01, f"got {f['midpoint']}")

summarise_fvgs(fvgs)

# -- TEST 2: Bearish FVG detection ---------------------------------------------
print(f"\n{BOLD}-- TEST 2: Bearish FVG Detection --{RESET}")

# Pattern:
#   candle[-3]: low = 2490  (candle before gap)
#   candle[-2]: huge bearish impulse
#   candle[-1]: high = 2470  → gap = 2470 to 2490
rows = flat_candles(30) + [
    (2500, 2502, 2490, 2492),   # i-2: low = 2490
    (2490, 2491, 2440, 2445),   # i-1: impulse candle down
    (2445, 2470, 2438, 2442),   # i:   high = 2470  → FVG zone: 2470-2490
]
df_bear = make_df(rows)
fvgs = find_fvgs(df_bear, "bearish", lookback=10)

check("Bearish FVG found", len(fvgs) >= 1, f"found {len(fvgs)}")
if fvgs:
    f = fvgs[0]
    check("FVG type is bearish",        f["type"] == "bearish",  f"got {f['type']}")
    check("FVG high = 2490 (prev2 low)", abs(f["fvg_high"] - 2490) < 0.01, f"got {f['fvg_high']}")
    check("FVG low = 2470 (curr high)",  abs(f["fvg_low"]  - 2470) < 0.01, f"got {f['fvg_low']}")
    check("Gap size = 20",              abs(f["gap_size"] - 20) < 0.01,   f"got {f['gap_size']}")

summarise_fvgs(fvgs)

# -- TEST 3: No FVG when gap too small -----------------------------------------
print(f"\n{BOLD}-- TEST 3: Small Gap Filtered Out --{RESET}")

# Gap = 0.5 points - below ATR threshold
rows = flat_candles(30) + [
    (2500, 2500.3, 2499, 2500.2),  # i-2: high = 2500.3
    (2500, 2505,   2499, 2504),    # i-1: weak impulse
    (2503, 2505,   2500.8, 2504),  # i:   low = 2500.8 → gap = 0.5
]
df_small = make_df(rows)
fvgs = find_fvgs(df_small, "bullish", lookback=10, min_gap_atr_ratio=0.5)
check("Tiny FVG correctly filtered out", len(fvgs) == 0, f"found {len(fvgs)}")

# -- TEST 4: price_in_fvg ------------------------------------------------------
print(f"\n{BOLD}-- TEST 4: price_in_fvg --{RESET}")

fvg = {"type": "bullish", "fvg_low": 2510.0, "fvg_high": 2530.0,
       "gap_size": 20, "midpoint": 2520, "candle_index": 5, "timestamp": pd.Timestamp("2025-01-01")}

check("Price inside FVG (2520)",      price_in_fvg(2520.0, fvg))
check("Price at FVG bottom (2510)",   price_in_fvg(2510.0, fvg))
check("Price at FVG top (2530)",      price_in_fvg(2530.0, fvg))
check("Price below FVG (2505)",      not price_in_fvg(2505.0, fvg))
check("Price above FVG (2535)",      not price_in_fvg(2535.0, fvg))
check("None FVG returns False",      not price_in_fvg(2520.0, None))

# -- TEST 5: fvg_has_been_mitigated --------------------------------------------
print(f"\n{BOLD}-- TEST 5: FVG Mitigation Detection --{RESET}")

# Build a df where:
#  - FVG forms at index 32 (impulse candle)
#  - FVG zone: 2510-2530
#  - candles after index 32 trade BELOW 2510 → mitigated
rows_mitigated = flat_candles(30) + [
    (2500, 2510, 2498, 2508),   # i-2
    (2510, 2560, 2509, 2555),   # i-1: impulse (index 31)
    (2555, 2565, 2530, 2562),   # i  : low=2530, gap=2510-2530 (index 32)
    (2560, 2562, 2505, 2510),   # FILL candle - low=2505 < 2510 → mitigates FVG
]
df_mit = make_df(rows_mitigated)
fvg_mit = {
    "type": "bullish", "fvg_low": 2510.0, "fvg_high": 2530.0,
    "gap_size": 20, "midpoint": 2520, "candle_index": 31,
    "timestamp": df_mit.index[31]
}
check("Mitigated FVG detected",   fvg_has_been_mitigated(df_mit, fvg_mit))

# Same setup but no fill candle → not mitigated
rows_not_mitigated = flat_candles(30) + [
    (2500, 2510, 2498, 2508),
    (2510, 2560, 2509, 2555),   # impulse at index 31
    (2555, 2565, 2530, 2562),   # gap at index 32 - nothing after
]
df_not_mit = make_df(rows_not_mitigated)
fvg_not_mit = {
    "type": "bullish", "fvg_low": 2510.0, "fvg_high": 2530.0,
    "gap_size": 20, "midpoint": 2520, "candle_index": 31,
    "timestamp": df_not_mit.index[31]
}
check("Unmitigated FVG correctly returns False", not fvg_has_been_mitigated(df_not_mit, fvg_not_mit))
check("None FVG returns False",                  not fvg_has_been_mitigated(df_not_mit, None))

# -- TEST 6: get_latest_fvg excludes mitigated ---------------------------------
print(f"\n{BOLD}-- TEST 6: get_latest_fvg Excludes Mitigated --{RESET}")

# Reuse df_mit - the FVG has been filled, so get_latest_fvg should return None
# We add extra flat candles after the fill so the entire lookback window
# contains only mitigated or non-existent FVGs
rows_all_mitigated = flat_candles(30) + [
    (2500, 2510, 2498, 2508),   # i-2
    (2510, 2560, 2509, 2555),   # i-1: impulse (index 31)
    (2555, 2565, 2530, 2562),   # i  : gap 2510-2530 (index 32)
    (2560, 2562, 2500, 2510),   # fill: low=2500 < 2510 - mitigates FVG
    (2510, 2515, 2500, 2512),   # more candles below 2510 to clear any secondary gaps
    (2510, 2514, 2499, 2511),
]
df_all_mit = make_df(rows_all_mitigated)
result = get_latest_fvg(df_all_mit, "bullish", lookback=10, exclude_mitigated=True)
check("Mitigated FVG excluded by get_latest_fvg", result is None, f"got {result}")

result_no_excl = get_latest_fvg(df_mit, "bullish", lookback=10, exclude_mitigated=False)
check("Mitigated FVG returned when exclude_mitigated=False", result_no_excl is not None)

# -- TEST 7: price_approaching_fvg ---------------------------------------------
print(f"\n{BOLD}-- TEST 7: price_approaching_fvg --{RESET}")

fvg_b = {"type": "bullish", "fvg_low": 2510.0, "fvg_high": 2530.0,
          "gap_size": 20, "midpoint": 2520, "candle_index": 5,
          "timestamp": pd.Timestamp("2025-01-01")}

# 0.2% of 2533 ≈ 5.07 - price 2533 is above fvg_high (2530) and within ~3pts
check("Approaching bullish FVG from above (2533)", price_approaching_fvg(2533.0, fvg_b))
check("Not approaching - too far above (2545)",   not price_approaching_fvg(2545.0, fvg_b))
check("Inside FVG is not 'approaching' (2520)",   not price_approaching_fvg(2520.0, fvg_b))

fvg_s = {"type": "bearish", "fvg_low": 2470.0, "fvg_high": 2490.0,
          "gap_size": 20, "midpoint": 2480, "candle_index": 5,
          "timestamp": pd.Timestamp("2025-01-01")}

check("Approaching bearish FVG from below (2468)", price_approaching_fvg(2468.0, fvg_s))
check("Not approaching - too far below (2450)",    not price_approaching_fvg(2450.0, fvg_s))

# -- TEST 8: Edge cases --------------------------------------------------------
print(f"\n{BOLD}-- TEST 8: Edge Cases --{RESET}")

check("Empty df returns []",       find_fvgs(make_df([]), "bullish") == [])
check("2-row df returns []",       find_fvgs(make_df(flat_candles(2)), "bullish") == [])
check("None df returns []",        find_fvgs(None, "bullish") == [])
check("get_latest_fvg on None",    get_latest_fvg(None, "bullish") is None)

try:
    find_fvgs(df_bull, "sideways")
    check("Invalid direction raises ValueError", False)
except ValueError:
    check("Invalid direction raises ValueError", True)

# -- SUMMARY -------------------------------------------------------------------
print(f"\n{BOLD}{'='*50}{RESET}")
total = passed + failed
if failed == 0:
    print(f"{BOLD}{GREEN}  ALL {total} TESTS PASSED{RESET}")
    print(f"{GREEN}  fvg_detector.py is ready.{RESET}")
else:
    print(f"{BOLD}{RED}  {failed}/{total} TESTS FAILED{RESET}")
print(f"{BOLD}{'='*50}{RESET}\n")

sys.exit(0 if failed == 0 else 1)