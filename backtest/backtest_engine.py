# backtest/backtest_engine.py

import pandas as pd
from backtesting import Backtest, Strategy

from strategy.entry_logic import check_entry_signal
from strategy.risk_management import build_trade_parameters
from indicators.atr import calculate_atr
from strategy.trailing_stop import update_trailing_stop
from backtest.trade_logger import TradeLogger
from strategy.reason_codes import LONG_ENTRY, SHORT_ENTRY



class GoldSMCStrategy(Strategy):
    # ---- Strategy parameters (tunable) ----
    risk_percent = 0.01
    atr_period = 14
    atr_multiplier = 2.0
    reward_risk_ratio = 2.0
    obv_ma_period = 10
    pivot_lookback = 2

def init(self):
    # ATR on 1H for SL sizing
    self.atr_1h = self.I(
        lambda h, l, c: calculate_atr(
            pd.DataFrame({"high": h, "low": l, "close": c}),
            self.atr_period
        ),
        self.data.High,
        self.data.Low,
        self.data.Close
    )

    # Trade logger
    self.trade_logger = TradeLogger("trade_journal.csv")


    def next(self):

        # ---------------------------
        # TRAILING STOP MANAGEMENT
        # ---------------------------
        if self.position:
            current_price = self.data.Close[-1]
            atr_value = self.atr_1h[-1]

            trade = self.position

            new_sl = update_trailing_stop(
                direction="long" if trade.is_long else "short",
                entry_price=trade.entry_price,
                current_price=current_price,
                atr_value=atr_value,
                current_stop=trade.sl
            )

            # Only update if SL tightens
            if new_sl != trade.sl:
                trade.sl = new_sl

            return  # do NOT open new trades while one is active

        # --- Build dataframes ---
        df_1h = self.data.df.copy()

        df_4h = (
            df_1h
            .resample("4H")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            })
            .dropna()
            .rename(columns=str.lower)
        )

        df_1d = (
            df_1h
            .resample("1D")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            })
            .dropna()
            .rename(columns=str.lower)
        )

        df_1h = df_1h.rename(columns=str.lower)

        # --- Entry Signal ---
        signal = check_entry_signal(
            df_1h=df_1h,
            df_4h=df_4h,
            df_1d=df_1d,
            atr_period=self.atr_period,
            obv_ma_period=self.obv_ma_period,
            pivot_lookback=self.pivot_lookback
        )

        if signal is None:
            return

        entry_price = self.data.Close[-1]
        atr_value = self.atr_1h[-1]

        # --- Risk Management ---
        trade = build_trade_parameters(
            account_balance=self.equity,
            risk_percent=self.risk_percent,
            entry_price=entry_price,
            atr_value=atr_value,
            direction=signal,
            pip_value=1.0,  # simplified for backtest
            atr_multiplier=self.atr_multiplier,
            reward_risk_ratio=self.reward_risk_ratio
        )

        if trade["position_size"] <= 0:
            return

        # --- Execute trade ---
        if signal == "long":
         self.trade_logger.log_trade(
        direction="long",
        entry_price=entry_price,
        stop_loss=trade["stop_loss"],
        take_profit=trade["take_profit"],
        position_size=trade["position_size"],
        equity=self.equity,
        reason=LONG_ENTRY
    )

         self.buy(
        size=trade["position_size"],
        sl=trade["stop_loss"],
        tp=trade["take_profit"]
    )


        if signal == "short":
         self.trade_logger.log_trade(
        direction="short",
        entry_price=entry_price,
        stop_loss=trade["stop_loss"],
        take_profit=trade["take_profit"],
        position_size=trade["position_size"],
        equity=self.equity,
        reason=SHORT_ENTRY
    )

         self.sell(
        size=trade["position_size"],
        sl=trade["stop_loss"],
        tp=trade["take_profit"]
    )
