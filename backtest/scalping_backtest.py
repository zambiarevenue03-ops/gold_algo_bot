"""
backtest/scalping_backtest.py
==============================
Backtesting engine for the XAUUSD scalping bot.

How it works:
  1. Walk forward candle-by-candle through 15M data
  2. At each candle, build 4H and 1H slices (resampled from 15M)
  3. Call check_entry_signal_detailed() — the same function live trading uses
  4. On signal: build trade parameters, then scan FORWARD for SL/TP hits
  5. Record every trade outcome to a detailed journal

Output metrics:
  - Win rate, expectancy, avg R, max drawdown
  - Rejection reason frequency table (shows what's blocking signals)
  - Equity curve data
  - Per-trade CSV journal

Design principles:
  - Zero lookahead bias: at bar[i], only data up to bar[i] is used
  - Signal spacing: minimum gap between consecutive entries
  - Daily risk guard: same kill switches as live trading
  - Regime awareness: results split by date ranges

Usage:
    from backtest.scalping_backtest import ScalpBacktest, BacktestParams
    bt = ScalpBacktest(df_15m, params=BacktestParams())
    results = bt.run()
    bt.print_summary(results)
    bt.save_journal(results, "backtest/results/scalp_journal.csv")
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scalping.entry_logic import check_entry_signal_detailed, RejectionTracker, Rejection
from scalping.risk_model   import (
    ScalpRiskParams, build_trade_parameters,
    DailyRiskGuard, DEFAULT_PARAMS,
)


# ── backtest configuration ─────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    """All backtest settings in one place."""
    # Account
    initial_balance:     float = 10_000.0
    risk_pct:            float = 0.0025    # 0.25% per trade

    # Signal generation
    warmup_candles:      int   = 300       # candles before first signal attempt
    step:                int   = 1         # evaluate every N candles (1=every, 5=faster)
    min_signal_gap_candles: int = 24       # min 15M candles between signals (24×15=6hrs)

    # HTF resample windows (in 15M candles)
    candles_1h:          int   = 4         # 4 × 15M = 1H
    candles_4h:          int   = 16        # 16 × 15M = 4H
    min_1h_candles:      int   = 60        # min 1H candles needed
    min_4h_candles:      int   = 30        # min 4H candles needed

    # Trade simulation
    commission_per_lot:  float = 7.0       # USD round-trip per lot
    spread_points:       float = 0.3       # XAUUSD typical spread
    slippage_points:     float = 0.2       # execution slippage

    # Risk guard
    max_daily_loss_pct:  float = 0.02
    max_weekly_loss_pct: float = 0.05
    max_consecutive_losses: int = 3

    # Entry logic passthrough
    pivot_lookback:      int   = 2
    obv_ma_period:       int   = 10
    atr_period:          int   = 14
    atr_ma_period:       int   = 14
    fvg_lookback:        int   = 50
    disp_lookback:       int   = 30
    choch_max_candles:   int   = 15
    min_soft_score:      int   = 1
    session_filter:      bool  = True
    utc_offset_hours:    int   = 0

    # Risk model passthrough
    risk_params:         ScalpRiskParams = field(default_factory=ScalpRiskParams)


# ── resample helper ────────────────────────────────────────────────────────────

def _resample_ohlcv(df_15m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 15M data to a higher timeframe."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return df_15m.resample(rule).agg(agg).dropna()


# ── trade simulator ────────────────────────────────────────────────────────────

def _simulate_trade(
    df_15m:   pd.DataFrame,
    entry_bar: int,
    trade:    dict,
    params:   BacktestParams,
) -> dict:
    """
    Walk forward from entry_bar to find SL or TP hits.
    
    Returns a result dict with outcome details.
    Partial exits are modelled: 50% at TP1, 30% at TP2, 20% at TP3/SL.
    """
    signal   = trade["signal"]
    entry    = trade["entry"] + params.spread_points + params.slippage_points
    sl       = trade["stop_loss"]
    tp1, tp2, tp3 = trade["tp1"], trade["tp2"], trade["tp3"]
    lot      = trade["lot_size"]
    risk     = trade["risk"]

    tp1_pct, tp2_pct, tp3_pct = trade["tp1_pct"], trade["tp2_pct"], trade["tp3_pct"]
    pip      = params.risk_params.pip_size
    pv       = params.risk_params.point_value

    # Track state
    tp1_hit = tp2_hit = False
    remaining_lot = lot
    total_pnl = 0.0
    exit_bar  = entry_bar
    exit_price = entry
    outcome   = "open"

    future = df_15m.iloc[entry_bar + 1:]

    for i, (ts, row) in enumerate(future.iterrows()):
        bar_high = row["high"]
        bar_low  = row["low"]

        if signal == "long":
            # Check TP1
            if not tp1_hit and bar_high >= tp1:
                lot1 = round(lot * tp1_pct, 2)
                pnl1 = (tp1 - entry) / pip * pv * lot1
                total_pnl    += pnl1
                remaining_lot = round(remaining_lot - lot1, 2)
                tp1_hit       = True

            # Check TP2
            if tp1_hit and not tp2_hit and bar_high >= tp2:
                lot2 = round(lot * tp2_pct, 2)
                pnl2 = (tp2 - entry) / pip * pv * lot2
                total_pnl    += pnl2
                remaining_lot = round(remaining_lot - lot2, 2)
                tp2_hit       = True

            # Check TP3 / runner exit
            if tp2_hit and bar_high >= tp3:
                pnl3 = (tp3 - entry) / pip * pv * remaining_lot
                total_pnl += pnl3
                exit_bar   = entry_bar + 1 + i
                exit_price = tp3
                outcome    = "tp3"
                break

            # Check SL
            if bar_low <= sl:
                exit_price = sl
                if tp1_hit:
                    # SL hit after TP1 — partial win
                    pnl_sl = (sl - entry) / pip * pv * remaining_lot
                    total_pnl += pnl_sl
                    outcome = "sl_after_tp1" if not tp2_hit else "sl_after_tp2"
                else:
                    # Full loss
                    pnl_sl = (sl - entry) / pip * pv * lot
                    total_pnl = pnl_sl
                    outcome = "sl"
                exit_bar = entry_bar + 1 + i
                break

        else:  # short
            if not tp1_hit and bar_low <= tp1:
                lot1 = round(lot * tp1_pct, 2)
                pnl1 = (entry - tp1) / pip * pv * lot1
                total_pnl    += pnl1
                remaining_lot = round(remaining_lot - lot1, 2)
                tp1_hit       = True

            if tp1_hit and not tp2_hit and bar_low <= tp2:
                lot2 = round(lot * tp2_pct, 2)
                pnl2 = (entry - tp2) / pip * pv * lot2
                total_pnl    += pnl2
                remaining_lot = round(remaining_lot - lot2, 2)
                tp2_hit       = True

            if tp2_hit and bar_low <= tp3:
                pnl3 = (entry - tp3) / pip * pv * remaining_lot
                total_pnl += pnl3
                exit_bar   = entry_bar + 1 + i
                exit_price = tp3
                outcome    = "tp3"
                break

            if bar_high >= sl:
                exit_price = sl
                if tp1_hit:
                    pnl_sl = (entry - sl) / pip * pv * remaining_lot
                    total_pnl += pnl_sl
                    outcome = "sl_after_tp1" if not tp2_hit else "sl_after_tp2"
                else:
                    pnl_sl = (entry - sl) / pip * pv * lot
                    total_pnl = pnl_sl
                    outcome = "sl"
                exit_bar = entry_bar + 1 + i
                break
    else:
        # Reached end of data without resolution
        last_close = future["close"].iloc[-1] if len(future) > 0 else entry
        if signal == "long":
            total_pnl = (last_close - entry) / pip * pv * remaining_lot
        else:
            total_pnl = (entry - last_close) / pip * pv * remaining_lot
        exit_bar   = len(df_15m) - 1
        exit_price = last_close
        outcome    = "end_of_data"

    # Commission
    commission = params.commission_per_lot * lot
    net_pnl    = total_pnl - commission

    # R-multiple
    risk_usd   = (risk / pip) * pv * lot
    r_multiple = net_pnl / risk_usd if risk_usd > 0 else 0.0

    return {
        "outcome":    outcome,
        "net_pnl":    round(net_pnl, 2),
        "gross_pnl":  round(total_pnl, 2),
        "commission": round(commission, 2),
        "r_multiple": round(r_multiple, 3),
        "tp1_hit":    tp1_hit,
        "tp2_hit":    tp2_hit,
        "exit_price": round(exit_price, 2),
        "exit_bar":   exit_bar,
        "bars_held":  exit_bar - entry_bar,
    }


# ── main backtest class ────────────────────────────────────────────────────────

class ScalpBacktest:
    """
    Walk-forward backtest for the scalping bot.

    Parameters
    ----------
    df_15m  : pd.DataFrame — 15M OHLCV data (full history)
    params  : BacktestParams
    """

    def __init__(self, df_15m: pd.DataFrame, params: BacktestParams = None):
        self.df_15m  = df_15m.copy()
        self.params  = params or BacktestParams()

    def run(self) -> dict:
        """
        Execute the backtest. Returns a results dict with:
          - trades      : list of trade dicts
          - equity_curve: list of (timestamp, balance) tuples
          - rejections  : RejectionTracker
          - metrics     : computed performance metrics
        """
        p         = self.params
        df        = self.df_15m
        balance   = p.initial_balance
        trades    = []
        equity    = [(df.index[0], balance)]
        tracker   = RejectionTracker()

        guard = DailyRiskGuard(
            starting_balance    = p.initial_balance,
            max_daily_loss_pct  = p.max_daily_loss_pct,
            max_weekly_loss_pct = p.max_weekly_loss_pct,
            max_consecutive_losses = p.max_consecutive_losses,
        )

        last_signal_bar = -p.min_signal_gap_candles  # allow signal from start
        last_week       = None
        last_day        = None
        active_trade_until_bar = -1  # bar index when current trade resolves

        print(f"\n  Running backtest on {len(df):,} candles "
              f"({df.index[0].date()} → {df.index[-1].date()})...")
        print(f"  Warmup: {p.warmup_candles} candles, "
              f"Step: {p.step}, Min gap: {p.min_signal_gap_candles} bars\n")

        for i in range(p.warmup_candles, len(df), p.step):
            current_ts  = df.index[i]
            current_day = current_ts.date()

            # ── day / week reset ──────────────────────────────────────────────
            week_num = current_ts.isocalendar()[1]
            if last_week is None:
                last_week = week_num
            if last_day is None:
                last_day = current_day

            if current_day != last_day:
                guard.new_day(balance)
                last_day = current_day

            if week_num != last_week:
                guard.new_week(balance)
                last_week = week_num

            # ── skip if trade active ──────────────────────────────────────────
            if i <= active_trade_until_bar:
                continue

            # ── signal gap filter ─────────────────────────────────────────────
            if (i - last_signal_bar) < p.min_signal_gap_candles:
                continue

            # ── kill switch check ─────────────────────────────────────────────
            can_trade, _ = guard.can_trade(balance, current_ts)
            if not can_trade:
                continue

            # ── build HTF slices ──────────────────────────────────────────────
            slice_15m = df.iloc[:i + 1]

            # Resample to 1H and 4H
            df_1h = _resample_ohlcv(slice_15m, "1h")
            df_4h = _resample_ohlcv(slice_15m, "4h")

            if len(df_1h) < p.min_1h_candles or len(df_4h) < p.min_4h_candles:
                continue

            # ── entry signal check ────────────────────────────────────────────
            signal, reason, scores = check_entry_signal_detailed(
                df_1h  = df_1h,
                df_4h  = df_4h,
                df_15m = slice_15m,
                pivot_lookback        = p.pivot_lookback,
                obv_ma_period         = p.obv_ma_period,
                atr_period            = p.atr_period,
                atr_ma_period         = p.atr_ma_period,
                fvg_lookback          = p.fvg_lookback,
                disp_lookback         = p.disp_lookback,
                choch_max_candles_ago = p.choch_max_candles,
                min_soft_score        = p.min_soft_score,
                session_filter        = p.session_filter,
                utc_offset_hours      = p.utc_offset_hours,
            )
            tracker.record(reason)

            if signal is None:
                continue

            # ── build trade parameters ────────────────────────────────────────
            entry_price = float(df.iloc[i]["close"])
            atr_val     = float(slice_15m["high"].rolling(14).max().iloc[-1] -
                                slice_15m["low"].rolling(14).min().iloc[-1]) / 4

            trade = build_trade_parameters(
                signal          = signal,
                entry           = entry_price,
                fvg             = scores.get("fvg"),
                displacement    = scores.get("displacement"),
                atr             = max(atr_val, 2.0),
                account_balance = balance,
                params          = p.risk_params,
                timestamp       = current_ts,
            )

            if trade is None:
                continue

            # ── simulate trade ────────────────────────────────────────────────
            result = _simulate_trade(df, i, trade, p)

            # ── update state ──────────────────────────────────────────────────
            balance += result["net_pnl"]
            guard.record_trade(result["net_pnl"], current_ts)
            equity.append((current_ts, round(balance, 2)))

            last_signal_bar       = i
            active_trade_until_bar = result["exit_bar"]

            # ── log trade ─────────────────────────────────────────────────────
            trades.append({
                "entry_time":   current_ts,
                "exit_time":    df.index[min(result["exit_bar"], len(df)-1)],
                "signal":       signal,
                "entry_price":  trade["entry"],
                "stop_loss":    trade["stop_loss"],
                "tp1":          trade["tp1"],
                "tp2":          trade["tp2"],
                "tp3":          trade["tp3"],
                "lot_size":     trade["lot_size"],
                "risk_usd":     trade["risk_amount_usd"],
                "outcome":      result["outcome"],
                "net_pnl":      result["net_pnl"],
                "r_multiple":   result["r_multiple"],
                "tp1_hit":      result["tp1_hit"],
                "tp2_hit":      result["tp2_hit"],
                "bars_held":    result["bars_held"],
                "balance":      round(balance, 2),
                "bias":         scores.get("bias", ""),
                "soft_score":   scores.get("soft_score", 0),
            })

        # ── compute metrics ───────────────────────────────────────────────────
        metrics = self._compute_metrics(trades, p.initial_balance, balance)

        return {
            "trades":       trades,
            "equity_curve": equity,
            "rejections":   tracker,
            "metrics":      metrics,
            "final_balance": round(balance, 2),
        }

    def _compute_metrics(self, trades: list, initial: float, final: float) -> dict:
        if not trades:
            return {"total_trades": 0, "note": "No trades generated"}

        df = pd.DataFrame(trades)
        wins   = df[df["net_pnl"] > 0]
        losses = df[df["net_pnl"] <= 0]
        sl_only = df[df["outcome"] == "sl"]

        win_rate    = len(wins) / len(df) if len(df) > 0 else 0
        avg_win_r   = wins["r_multiple"].mean()   if len(wins) > 0 else 0
        avg_loss_r  = losses["r_multiple"].mean() if len(losses) > 0 else 0
        expectancy  = (win_rate * avg_win_r) + ((1 - win_rate) * avg_loss_r)

        # Max drawdown
        equity_vals = [e[1] for e in self.params.initial_balance and [(None, initial)] or []]
        balances    = [initial] + df["balance"].tolist()
        peak        = initial
        max_dd      = 0.0
        max_dd_pct  = 0.0
        for b in balances:
            if b > peak:
                peak = b
            dd     = peak - b
            dd_pct = dd / peak if peak > 0 else 0
            if dd_pct > max_dd_pct:
                max_dd     = dd
                max_dd_pct = dd_pct

        # Profit factor
        gross_profit = wins["net_pnl"].sum()   if len(wins) > 0 else 0
        gross_loss   = abs(losses["net_pnl"].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "total_trades":     len(df),
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate":         round(win_rate * 100, 1),
            "avg_win_r":        round(avg_win_r, 3),
            "avg_loss_r":       round(avg_loss_r, 3),
            "expectancy_r":     round(expectancy, 3),
            "profit_factor":    round(profit_factor, 2),
            "total_pnl":        round(final - initial, 2),
            "total_return_pct": round((final - initial) / initial * 100, 2),
            "max_drawdown":     round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct * 100, 2),
            "tp1_hit_rate":     round(df["tp1_hit"].mean() * 100, 1),
            "tp2_hit_rate":     round(df["tp2_hit"].mean() * 100, 1),
            "full_sl_rate":     round(len(sl_only) / len(df) * 100, 1),
            "avg_bars_held":    round(df["bars_held"].mean(), 1),
            "longs":            int((df["signal"] == "long").sum()),
            "shorts":           int((df["signal"] == "short").sum()),
        }

    def print_summary(self, results: dict) -> None:
        """Print a formatted backtest summary."""
        m  = results["metrics"]
        print(f"\n{'═'*55}")
        print(f"  SCALPING BACKTEST RESULTS")
        print(f"{'═'*55}")
        if m.get("total_trades", 0) == 0:
            print(f"  No trades generated.")
            print(f"\n  Rejection breakdown:")
            results["rejections"].summary()
            return

        print(f"  Period:         {self.df_15m.index[0].date()} → {self.df_15m.index[-1].date()}")
        print(f"  Total trades:   {m['total_trades']}  (Long: {m['longs']}, Short: {m['shorts']})")
        print(f"  Win rate:       {m['win_rate']}%")
        print(f"  Avg win R:      +{m['avg_win_r']:.2f}R")
        print(f"  Avg loss R:     {m['avg_loss_r']:.2f}R")
        print(f"  Expectancy:     {m['expectancy_r']:+.3f}R per trade")
        print(f"  Profit factor:  {m['profit_factor']:.2f}")
        print(f"  ─────────────────────────────────────────────")
        print(f"  TP1 hit rate:   {m['tp1_hit_rate']}%")
        print(f"  TP2 hit rate:   {m['tp2_hit_rate']}%")
        print(f"  Full SL rate:   {m['full_sl_rate']}%")
        print(f"  Avg bars held:  {m['avg_bars_held']} × 15M candles")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Net P&L:        ${m['total_pnl']:+,.2f}")
        print(f"  Total return:   {m['total_return_pct']:+.2f}%")
        print(f"  Max drawdown:   ${m['max_drawdown']:,.2f}  ({m['max_drawdown_pct']:.2f}%)")
        print(f"  Final balance:  ${results['final_balance']:,.2f}")
        print(f"{'═'*55}")
        print(f"\n  Signal rejection breakdown:")
        results["rejections"].summary()

    def save_journal(self, results: dict, path: str) -> None:
        """Save trade journal to CSV."""
        if not results["trades"]:
            print("  No trades to save.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(results["trades"])
        df.to_csv(path, index=False)
        print(f"\n  Journal saved: {path}  ({len(df)} trades)")

    def save_equity_curve(self, results: dict, path: str) -> None:
        """Save equity curve to CSV."""
        ec = results["equity_curve"]
        if not ec:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(ec, columns=["timestamp", "balance"])
        df.to_csv(path, index=False)
        print(f"  Equity curve saved: {path}")