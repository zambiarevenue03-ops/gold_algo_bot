import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scalping.displacement import (
    is_displacement,
    displacement_strength,
    find_displacements,
    get_latest_displacement,
    displacement_preceded_fvg,
    summarise_displacements,
)

# -- colour helpers -------------------------------------------------------------
GREEN = "\033[92m"; RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  {GREEN}OK{RESET}  {name}")
        passed += 1
    else:
        print(f"  {RED}X{RESET}  {RED}{name}{RESET}" + (f"  <- {detail}" if detail else ""))
        failed += 1


# -- data helpers ---------------------------------------------------------------

def make_df(rows: list) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(rows), freq="15min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=dates)
    df["volume"] = 1000
    return df


def quiet_candles(n, base=2500.0):
    """Small-range, roughly flat candles to establish a baseline avg body."""
    result = []
    for i in range(n):
        o = base + (i % 3) * 0.5
        result.append((o, o + 3.0, o - 3.0, o + 1.0))  # body=1, range=6
    return result


def make_candle(o, h, l, c):
    return pd.Series({"open": o, "high": h, "low": l, "close": c, "volume": 1000})


# -- TEST 1: is_displacement - bullish -----------------------------------------
print(f"\n{BOLD}-- TEST 1: is_displacement - Bullish --{RESET}")

# avg_body = 1.0, avg_range = 6.0
# Valid bullish displacement: big body, closes near high
c_bull = make_candle(2500, 2530, 2498, 2528)
# body=28, range=32, body_ratio=0.875, close_pos=(2528-2498)/32=0.9375
check("Valid bullish displacement accepted",
      is_displacement(c_bull, avg_body=1.0, avg_range=6.0, direction="bullish"))

# Bearish candle - should not qualify as bullish
c_bear_candle = make_candle(2528, 2530, 2498, 2502)
check("Bearish candle rejected as bullish",
      not is_displacement(c_bear_candle, avg_body=1.0, avg_range=6.0, direction="bullish"))

# Small body - below body_atr_ratio threshold
c_small = make_candle(2500, 2502, 2499, 2501)
# body=1, avg_body=1 → needs 1.5*1=1.5 → fails
check("Small body rejected (below body_atr_ratio)",
      not is_displacement(c_small, avg_body=1.0, avg_range=6.0, direction="bullish"))

# Large body but mostly wicks (doji-like with big wicks)
c_wick = make_candle(2500, 2540, 2460, 2510)
# body=10, range=80, body_ratio=0.125 → below min_body_ratio 0.55
check("Wick-heavy candle rejected (low body ratio)",
      not is_displacement(c_wick, avg_body=1.0, avg_range=6.0, direction="bullish"))

# Closes too low - not bullish conviction
c_low_close = make_candle(2500, 2530, 2498, 2506)
# body=6, range=32, body_ratio=0.1875 → fails body ratio
check("Bullish candle closing near open rejected",
      not is_displacement(c_low_close, avg_body=1.0, avg_range=6.0, direction="bullish"))


# -- TEST 2: is_displacement - bearish -----------------------------------------
print(f"\n{BOLD}-- TEST 2: is_displacement - Bearish --{RESET}")

# Valid bearish displacement: big body, closes near low
c_bear = make_candle(2530, 2532, 2498, 2502)
# body=28, range=34, body_ratio=0.82, close_pos=(2502-2498)/34=0.118 → bearish conviction OK
check("Valid bearish displacement accepted",
      is_displacement(c_bear, avg_body=1.0, avg_range=6.0, direction="bearish"))

# Bullish candle - should not qualify as bearish
check("Bullish candle rejected as bearish",
      not is_displacement(c_bull, avg_body=1.0, avg_range=6.0, direction="bearish"))

# Bearish but closes too high
c_bear_weak = make_candle(2530, 2532, 2498, 2525)
# close_pos = (2525-2498)/34 = 0.79 → above (1-0.60)=0.40 → fails conviction
check("Bearish candle closing near open rejected",
      not is_displacement(c_bear_weak, avg_body=1.0, avg_range=6.0, direction="bearish"))


# -- TEST 3: is_displacement - edge cases --------------------------------------
print(f"\n{BOLD}-- TEST 3: is_displacement - Edge Cases --{RESET}")

check("avg_body=0 returns False",
      not is_displacement(c_bull, avg_body=0.0, avg_range=6.0, direction="bullish"))
check("avg_range=0 returns False",
      not is_displacement(c_bull, avg_body=1.0, avg_range=0.0, direction="bullish"))

# Zero-range candle
c_zero = make_candle(2500, 2500, 2500, 2500)
check("Zero-range candle returns False",
      not is_displacement(c_zero, avg_body=1.0, avg_range=6.0, direction="bullish"))

try:
    is_displacement(c_bull, 1.0, 6.0, "sideways")
    check("Invalid direction raises ValueError", False)
except ValueError:
    check("Invalid direction raises ValueError", True)


# -- TEST 4: displacement_strength ---------------------------------------------
print(f"\n{BOLD}-- TEST 4: displacement_strength --{RESET}")

# Perfect bullish: huge body, closes exactly at high, 4* avg body
c_perfect = make_candle(2500, 2540, 2499, 2540)
s_perfect = displacement_strength(c_perfect, avg_body=10.0, avg_range=20.0)
check("Perfect bullish strength close to 1.0", s_perfect > 0.85, f"got {s_perfect}")

# Weak candle: body barely above avg, lots of wicks
c_weak = make_candle(2500, 2530, 2498, 2515)
s_weak = displacement_strength(c_weak, avg_body=10.0, avg_range=20.0)
check("Weak candle has lower strength than perfect", s_weak < s_perfect, f"weak={s_weak}, perfect={s_perfect}")

check("Strength score is between 0 and 1",
      0.0 <= s_perfect <= 1.0 and 0.0 <= s_weak <= 1.0)

check("avg_body=0 returns 0.0",
      displacement_strength(c_bull, avg_body=0.0, avg_range=6.0) == 0.0)


# -- TEST 5: find_displacements ------------------------------------------------
print(f"\n{BOLD}-- TEST 5: find_displacements --{RESET}")

# 25 quiet candles (body≈1, range≈6) then 1 big bullish displacement
rows = quiet_candles(25) + [
    (2500, 2535, 2498, 2533),   # displacement: body=33, range=37, ratio=0.89
]
df_bull = make_df(rows)

disps = find_displacements(df_bull, "bullish", lookback=10)
check("Bullish displacement found in sequence", len(disps) >= 1, f"found {len(disps)}")
if disps:
    d = disps[0]
    check("Displacement direction is bullish",   d["direction"] == "bullish")
    check("Displacement body > avg_body * 1.5",  d["body"] > d["avg_body"] * 1.5,
          f"body={d['body']:.2f}, avg*1.5={d['avg_body']*1.5:.2f}")
    check("Displacement body_ratio >= 0.55",     d["body_ratio"] >= 0.55, f"got {d['body_ratio']:.3f}")
    check("Strength score in 0-1",              0.0 <= d["strength"] <= 1.0)

summarise_displacements(disps)

# Bearish displacement
rows_bear = quiet_candles(25) + [
    (2535, 2537, 2498, 2501),   # displacement: body=34, range=39, ratio=0.87
]
df_bear = make_df(rows_bear)

disps_bear = find_displacements(df_bear, "bearish", lookback=10)
check("Bearish displacement found", len(disps_bear) >= 1, f"found {len(disps_bear)}")

# No displacement when candles are all quiet
df_quiet = make_df(quiet_candles(30))
disps_quiet = find_displacements(df_quiet, "bullish", lookback=15)
check("No displacement in quiet market", len(disps_quiet) == 0, f"found {len(disps_quiet)}")


# -- TEST 6: get_latest_displacement -------------------------------------------
print(f"\n{BOLD}-- TEST 6: get_latest_displacement --{RESET}")

result = get_latest_displacement(df_bull, "bullish", lookback=10)
check("get_latest_displacement returns dict",   result is not None)
check("Returns most recent displacement",        result["timestamp"] == df_bull.index[-1])

result_none = get_latest_displacement(df_quiet, "bullish", lookback=15)
check("Returns None when no displacement",       result_none is None)

# min_strength filter
result_high_strength = get_latest_displacement(df_bull, "bullish", lookback=10, min_strength=0.99)
check("min_strength=0.99 filters out moderate displacement",
      result_high_strength is None,
      f"got strength={result['strength'] if result else 'N/A'}")

result_low_strength = get_latest_displacement(df_bull, "bullish", lookback=10, min_strength=0.0)
check("min_strength=0.0 does not filter anything", result_low_strength is not None)

check("None df returns None",  get_latest_displacement(None, "bullish") is None)


# -- TEST 7: displacement_preceded_fvg ----------------------------------------
print(f"\n{BOLD}-- TEST 7: displacement_preceded_fvg --{RESET}")

# Build a sequence: quiet → displacement → FVG candle sequence
# Displacement at index 25, FVG impulse at index 25 (same candle often creates both)
rows_linked = quiet_candles(24) + [
    (2500, 2510, 2498, 2508),   # index 24: pre-impulse (prev2 for FVG)
    (2508, 2545, 2507, 2543),   # index 25: displacement + impulse candle
    (2543, 2548, 2525, 2546),   # index 26: post-impulse (curr for FVG, low=2525)
]
# FVG zone: prev2_high(2510) to curr_low(2525)
df_linked = make_df(rows_linked)

fvg_dict = {
    "type":         "bullish",
    "fvg_low":      2510.0,
    "fvg_high":     2525.0,
    "gap_size":     15.0,
    "midpoint":     2517.5,
    "candle_index": 25,    # impulse candle
    "timestamp":    df_linked.index[25],
}

check("Displacement preceded FVG (should be True)",
      displacement_preceded_fvg(df_linked, fvg_dict, "bullish"))

check("None FVG returns False",
      not displacement_preceded_fvg(df_linked, None, "bullish"))
check("None df returns False",
      not displacement_preceded_fvg(None, fvg_dict, "bullish"))

# FVG with no displacement before it (all quiet candles)
fvg_no_disp = {
    "type":         "bullish",
    "fvg_low":      2503.0,
    "fvg_high":     2504.5,
    "gap_size":     1.5,
    "midpoint":     2503.75,
    "candle_index": 10,
    "timestamp":    df_quiet.index[10],
}
check("No displacement before quiet FVG returns False",
      not displacement_preceded_fvg(df_quiet, fvg_no_disp, "bullish"))


# -- TEST 8: Edge cases ---------------------------------------------------------
print(f"\n{BOLD}-- TEST 8: Edge Cases --{RESET}")

check("find_displacements on None returns []",
      find_displacements(None, "bullish") == [])
check("find_displacements on tiny df returns []",
      find_displacements(make_df(quiet_candles(5)), "bullish") == [])

try:
    find_displacements(df_bull, "sideways")
    check("Invalid direction raises ValueError", False)
except ValueError:
    check("Invalid direction raises ValueError", True)


# -- SUMMARY -------------------------------------------------------------------
print(f"\n{BOLD}{'='*50}{RESET}")
total = passed + failed
if failed == 0:
    print(f"{BOLD}{GREEN}  ALL {total} TESTS PASSED{RESET}")
    print(f"{GREEN}  displacement.py is ready.{RESET}")
else:
    print(f"{BOLD}{RED}  {failed}/{total} TESTS FAILED{RESET}")
print(f"{BOLD}{'='*50}{RESET}\n")

sys.exit(0 if failed == 0 else 1)