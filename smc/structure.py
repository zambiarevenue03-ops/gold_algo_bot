from typing import List, Tuple, Optional


def determine_trend(swing_highs: List[int], swing_lows: List[int]) -> Optional[str]:
    """Determine a simple trend label from swing high/low indices.

    Returns:
        "bullish" if higher highs / higher lows pattern detected,
        "bearish" if lower highs / lower lows pattern detected,
        None if unclear.
    """
    if not swing_highs or not swing_lows:
        return None

    # Simple heuristic: compare last two swing highs/lows when available
    try:
        last_highs = swing_highs[-2:]
        last_lows = swing_lows[-2:]
    except Exception:
        return None

    if len(last_highs) < 2 or len(last_lows) < 2:
        return None

    # If the most recent high is after the previous and the most recent low is
    # after the previous, assume bullish; the converse implies bearish.
    if last_highs[-1] > last_highs[-2] and last_lows[-1] > last_lows[-2]:
        return "bullish"
    if last_highs[-1] < last_highs[-2] and last_lows[-1] < last_lows[-2]:
        return "bearish"

    return None
