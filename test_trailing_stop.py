"""
test_trailing_stop.py
======================
Tests for scalping/trailing_stop.py

Run from gold_algo_bot/ root:
    python test_trailing_stop.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scalping.trailing_stop import (
    calculate_trailing_stop,
    TrailingStopManager,
)

GREEN = "\033[92m"; RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"
passed = 0; failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  {GREEN}✓{RESET}  {name}"); passed += 1
    else:
        print(f"  {RED}✗{RESET}  {RED}{name}{RESET}" + (f"  ← {detail}" if detail else "")); failed += 1


# ── TEST 1: calculate_trailing_stop — Phase 1 (no trail) ─────────────────────
print(f"\n{BOLD}── TEST 1: Phase 1 — No Trailing Before TP1 ──{RESET}")

sl_long = calculate_trailing_stop(
    signal="long", current_price=2510, original_sl=2490, entry_price=2500,
    tp1=2515, tp2=2525, tp1_hit=False, tp2_hit=False, atr=8.0
)
check("Long Phase 1: SL unchanged", sl_long == 2490, f"got {sl_long}")

sl_short = calculate_trailing_stop(
    signal="short", current_price=2510, original_sl=2530, entry_price=2520,
    tp1=2505, tp2=2495, tp1_hit=False, tp2_hit=False, atr=8.0
)
check("Short Phase 1: SL unchanged", sl_short == 2530, f"got {sl_short}")


# ── TEST 2: Phase 2 — Breakeven After TP1 ────────────────────────────────────
print(f"\n{BOLD}── TEST 2: Phase 2 — Breakeven After TP1 ──{RESET}")

sl_be = calculate_trailing_stop(
    signal="long", current_price=2520, original_sl=2490, entry_price=2500,
    tp1=2515, tp2=2525, tp1_hit=True, tp2_hit=False, atr=8.0
)
check("Long Phase 2: SL moved to breakeven", sl_be == 2500, f"got {sl_be}")

# If price went up a lot after TP1 but before TP2, still just breakeven
sl_be2 = calculate_trailing_stop(
    signal="long", current_price=2523, original_sl=2490, entry_price=2500,
    tp1=2515, tp2=2525, tp1_hit=True, tp2_hit=False, atr=8.0
)
check("Long Phase 2: SL stays at breakeven even if price higher", sl_be2 == 2500)


# ── TEST 3: Phase 3 — Trailing After TP2 ─────────────────────────────────────
print(f"\n{BOLD}── TEST 3: Phase 3 — Trailing After TP2 ──{RESET}")

# TP2 hit, price at 2530, ATR=8, trail=0.5×8=4 → SL=2530-4=2526
sl_trail = calculate_trailing_stop(
    signal="long", current_price=2530, original_sl=2490, entry_price=2500,
    tp1=2515, tp2=2525, tp1_hit=True, tp2_hit=True, atr=8.0, trail_distance_atr=0.5
)
check("Long Phase 3: Trailing started", sl_trail == 2526, f"got {sl_trail}")

# Price moves higher to 2535 → trail follows to 2531
sl_trail2 = calculate_trailing_stop(
    signal="long", current_price=2535, original_sl=2490, entry_price=2500,
    tp1=2515, tp2=2525, tp1_hit=True, tp2_hit=True, atr=8.0, trail_distance_atr=0.5
)
check("Long Phase 3: Trail follows price up", sl_trail2 == 2531, f"got {sl_trail2}")

# Price reverses to 2525 → SL doesn't move back down (stays at 2531)
sl_trail3 = calculate_trailing_stop(
    signal="long", current_price=2525, original_sl=2490, entry_price=2500,
    tp1=2515, tp2=2525, tp1_hit=True, tp2_hit=True, atr=8.0, trail_distance_atr=0.5
)
# Expected: max(2525-4, 2500) = 2521, but prev SL was 2531 so it locks at entry minimum
check("Long Phase 3: SL never goes below breakeven", sl_trail3 >= 2500, f"got {sl_trail3}")


# ── TEST 4: TrailingStopManager — Full Trade Lifecycle ───────────────────────
print(f"\n{BOLD}── TEST 4: TrailingStopManager — Full Trade ──{RESET}")

manager = TrailingStopManager(
    signal="long", entry_price=2500, stop_loss=2490,
    tp1=2515, tp2=2525, tp3=2540, trail_distance_atr=0.5
)

check("Initial state: no TPs hit",      not manager.tp1_hit)
check("Initial SL = original",          manager.current_sl == 2490)
check("Initial position = 100%",        manager.remaining_position_pct == 1.0)

# Price moves to 2510 (before TP1)
new_sl = manager.update(current_price=2510, atr=8.0)
check("Before TP1: SL unchanged",       new_sl == 2490)
check("Before TP1: TP1 not hit yet",    not manager.tp1_hit)

# Price hits TP1 at 2515
new_sl = manager.update(current_price=2515, atr=8.0)
check("TP1 hit: flag set",              manager.tp1_hit)
check("TP1 hit: SL moved to breakeven", new_sl == 2500)
check("TP1 hit: position reduced to 50%", manager.remaining_position_pct == 0.5)
check("TP1 hit: 1 exit recorded",       len(manager.exits) == 1)

# Price hits TP2 at 2525
new_sl = manager.update(current_price=2525, atr=8.0)
check("TP2 hit: flag set",              manager.tp2_hit)
check("TP2 hit: trailing started",      new_sl > 2500)  # should be ~2521
check("TP2 hit: position reduced to 20%", manager.remaining_position_pct == 0.2)

# Price runs to 2535, trail follows
new_sl = manager.update(current_price=2535, atr=8.0)
check("Trail follows price up",         new_sl > 2525, f"got {new_sl}")

# Price reverses, hits trail SL
is_stopped = manager.is_stopped_out(current_price=new_sl - 1)
check("Stopped out when price hits trail SL", is_stopped)


# ── TEST 5: Short Trade Trailing ─────────────────────────────────────────────
print(f"\n{BOLD}── TEST 5: Short Trade Trailing ──{RESET}")

manager_short = TrailingStopManager(
    signal="short", entry_price=2520, stop_loss=2540,
    tp1=2505, tp2=2495, tp3=2480
)

# TP1 hit
manager_short.update(current_price=2505, atr=8.0)
check("Short TP1 hit",                  manager_short.tp1_hit)
check("Short breakeven",                manager_short.current_sl == 2520)

# TP2 hit
new_sl_short = manager_short.update(current_price=2495, atr=8.0)
check("Short TP2 hit",                  manager_short.tp2_hit)
check("Short trailing started",         new_sl_short < 2520)

# Price drops further, trail follows down
new_sl_short2 = manager_short.update(current_price=2490, atr=8.0)
check("Short trail follows price down", new_sl_short2 < new_sl_short, 
      f"prev={new_sl_short}, new={new_sl_short2}")


# ── TEST 6: Edge Cases ────────────────────────────────────────────────────────
print(f"\n{BOLD}── TEST 6: Edge Cases ──{RESET}")

# Invalid signal
sl_invalid = calculate_trailing_stop(
    signal="sideways", current_price=2520, original_sl=2490, entry_price=2500,
    tp1=2515, tp2=2525, tp1_hit=False, tp2_hit=False, atr=8.0
)
check("Invalid signal returns original SL", sl_invalid == 2490)

# Zero ATR (shouldn't crash)
sl_zero_atr = calculate_trailing_stop(
    signal="long", current_price=2530, original_sl=2490, entry_price=2500,
    tp1=2515, tp2=2525, tp1_hit=True, tp2_hit=True, atr=0.0
)
check("Zero ATR doesn't crash",         isinstance(sl_zero_atr, float))

# TP2 hit but no TP1 hit (shouldn't happen, but handle gracefully)
manager_weird = TrailingStopManager(
    signal="long", entry_price=2500, stop_loss=2490,
    tp1=2515, tp2=2525, tp3=2540
)
manager_weird.tp2_hit = True  # manually set (shouldn't happen naturally)
new_sl_weird = manager_weird.update(current_price=2530, atr=8.0)
check("TP2 without TP1 handled gracefully", isinstance(new_sl_weird, float))


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'='*55}{RESET}")
total = passed + failed
if failed == 0:
    print(f"{BOLD}{GREEN}  ALL {total} TESTS PASSED{RESET}")
    print(f"{GREEN}  trailing_stop.py is ready.{RESET}")
else:
    print(f"{BOLD}{RED}  {failed}/{total} TESTS FAILED{RESET}")
print(f"{BOLD}{'='*55}{RESET}\n")

sys.exit(0 if failed == 0 else 1)