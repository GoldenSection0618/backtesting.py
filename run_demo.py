from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG, SMA


class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, self.n1)
        self.ma2 = self.I(SMA, price, self.n2)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


if __name__ == "__main__":
    bt = Backtest(
        GOOG,
        SmaCross,
        commission=0.002,
        exclusive_orders=True,
    )

    stats = bt.run()
    print(stats)

    bt.plot(
        filename="sma_cross_result.html",
        open_browser=False,
    )
    print("Plot saved to sma_cross_result.html")
