# smc/order_blocks.py

import pandas as pd
from typing import Optional, Dict


def find_order_block(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 50,
    impulse_factor: float = 1.5,
    max_base_candles: int = 3
) -> Optional[Dict]:
    """
    Detect the most recent valid order block.

    Returns:
        {
            'type': 'bullish' or 'bearish',
            'zone_low': float,
            'zone_high': float,
            'index': int
        }
    """

    df = df.copy()
    df['range'] = df['high'] - df['low']
    avg_range = df['range'].rolling(14).mean()

    for i in range(len(df) - 2, len(df) - lookback, -1):

        # Impulse candle
        candle_range = df['range'].iloc[i]
        if candle_range < impulse_factor * avg_range.iloc[i]:
            continue

        open_price = df['open'].iloc[i]
        close_price = df['close'].iloc[i]

        # Bullish impulse → look for bearish base
        if direction == 'bullish' and close_price > open_price:
            for j in range(1, max_base_candles + 1):
                base_idx = i - j
                if base_idx < 0:
                    break

                # Base candle must be bearish
                if df['close'].iloc[base_idx] < df['open'].iloc[base_idx]:
                    return {
                        'type': 'bullish',
                        'zone_low': df['low'].iloc[base_idx],
                        'zone_high': df['high'].iloc[base_idx],
                        'index': base_idx
                    }

        # Bearish impulse → look for bullish base
        if direction == 'bearish' and close_price < open_price:
            for j in range(1, max_base_candles + 1):
                base_idx = i - j
                if base_idx < 0:
                    break

                # Base candle must be bullish
                if df['close'].iloc[base_idx] > df['open'].iloc[base_idx]:
                    return {
                        'type': 'bearish',
                        'zone_low': df['low'].iloc[base_idx],
                        'zone_high': df['high'].iloc[base_idx],
                        'index': base_idx
                    }

    return None


def price_in_order_block(
    price: float,
    zone_low: float,
    zone_high: float,
    buffer: float = 0.0
) -> bool:
    """
    Check if price is inside order block zone.
    Optional buffer expands zone slightly.
    """

    return (zone_low - buffer) <= price <= (zone_high + buffer)
import pandas as pd
import os

def load_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    # Convert 'Time' or 'Date' to datetime if exists
    for col in ['time', 'Time', 'date', 'Date', 'datetime', 'Datetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
            break
            
    return df
def check_dataframe_integrity(df):
    required_columns = ['Open', 'High', 'Low', 'Close']
    
    # Check for required columns (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    for req in required_columns:
        if req.lower() not in cols:
            raise ValueError(f"Missing required column: {req}")
    
    # Check for NaN values
    if df.isnull().values.any():
        print("Warning: Missing values (NaN) detected in dataset.")
        
    # Check for logical consistency
    if not (df[cols['high']] >= df[cols['low']]).all():
        raise ValueError("Integrity Error: High price is lower than Low price in some rows.")
    
    print("Integrity check passed: Basic structure and data logic are sound.")
