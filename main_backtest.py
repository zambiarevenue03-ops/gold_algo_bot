import pandas as pd
from backtesting import Backtest

from backtest.backtest_engine import GoldSMCStrategy

# Load 1H data (CSV)
df = pd.read_csv(
    "data/processed/XAUUSD_1H.csv",
    parse_dates=["Date"],
    index_col="Date"
)

df.columns = [c.capitalize() for c in df.columns]

bt = Backtest(
    df,
    GoldSMCStrategy,
    cash=10_000,
    commission=0.0002,  # realistic
    exclusive_orders=True
)

stats = bt.run()
print(stats)

bt.plot()
