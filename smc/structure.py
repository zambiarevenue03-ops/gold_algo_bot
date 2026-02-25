# smc/structure.py

from typing import List, Optional


def determine_trend(
    swing_highs: List[int],
    swing_lows: List[int]
) -> Optional[str]:
    """
    Determine market trend based on swing structure.

    Returns:
        'bullish'
        'bearish'
        None (neutral / unclear)
    """

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    # Last two swings
    last_high, prev_high = swing_highs[-1], swing_highs[-2]
    last_low, prev_low = swing_lows[-1], swing_lows[-2]

    # Bullish: Higher High + Higher Low
    if last_high > prev_high and last_low > prev_low:
        return 'bullish'

    # Bearish: Lower High + Lower Low
    if last_high < prev_high and last_low < prev_low:
        return 'bearish'

    return None


def trend_strength(
    swing_highs: List[int],
    swing_lows: List[int]
) -> int:
    """
    Optional helper:
    Returns a simple trend strength score.

    +2 = strong bullish
    -2 = strong bearish
     0 = neutral / weak
    """

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 0

    score = 0

    if swing_highs[-1] > swing_highs[-2]:
        score += 1
    elif swing_highs[-1] < swing_highs[-2]:
        score -= 1

    if swing_lows[-1] > swing_lows[-2]:
        score += 1
    elif swing_lows[-1] < swing_lows[-2]:
        score -= 1

    return score
