# backtest/trade_logger.py

import csv
from datetime import datetime
from pathlib import Path


class TradeLogger:
    def __init__(self, filename="trade_journal.csv"):
        self.filename = Path(filename)
        self._init_file()

    def _init_file(self):
        if not self.filename.exists():
            with open(self.filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "direction",
                    "entry_price",
                    "stop_loss",
                    "take_profit",
                    "position_size",
                    "equity",
                    "reason"
                ])

    def log_trade(self, *, direction, entry_price, stop_loss,
                  take_profit, position_size, equity, reason):
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                direction,
                round(entry_price, 5),
                round(stop_loss, 5),
                round(take_profit, 5),
                round(position_size, 2),
                round(equity, 2),
                reason
            ])
