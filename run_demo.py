# ============================================================
# run_demo.py —— SMA 交叉策略最小运行示例
# ============================================================
# 上下文层：课程大作业的复现入口——本文件是验证工程可运行的最小脚本。
#           本脚本复现经典的 SMA（简单移动平均线）交叉策略。
# 功能层：加载 GOOG 日线，SMA(10)/SMA(20) 交叉产生买卖信号，
#          输出约 30 项回测统计并生成 sma_cross_result.html。
# 设计层：Strategy 基类用 ABCMeta 元类强制接口——模板方法模式。
#          self.I() 注册指标，crossover() 判断上穿。

from backtesting import Backtest, Strategy
from backtesting.lib import crossover      # 判断两条线是否上穿
from backtesting.test import GOOG, SMA     # GOOG: 日线数据, SMA: 移动平均函数


class SmaCross(Strategy):
    # 类变量也是策略参数，optimize() 可以自动搜索它们的最优值
    n1 = 10   # 快线周期
    n2 = 20   # 慢线周期

    def init(self):
        # init 在回测开始前只跑一次，在这里预计算全部指标
        price = self.data.Close
        self.ma1 = self.I(SMA, price, self.n1)   # 注册快线指标
        self.ma2 = self.I(SMA, price, self.n2)   # 注册慢线指标

    def next(self):
        # next 每根 K 线调用一次，self.data 只暴露到当前 bar
        if crossover(self.ma1, self.ma2):
            self.buy()           # 快线上穿慢线 → 做多
        elif crossover(self.ma2, self.ma1):
            self.sell()          # 快线下穿慢线 → 做空


if __name__ == "__main__":
    bt = Backtest(
        GOOG,                       # 2004-2013 Google 日线
        SmaCross,                   # 策略类（不是实例）
        commission=0.002,           # 佣金 0.2%，进场出场各扣一次
        exclusive_orders=True,      # 新订单自动平旧仓，同一时间最多一个仓位
    )
    stats = bt.run()
    print(stats)

    bt.plot(
        filename="sma_cross_result.html",
        open_browser=False,         # 不自动打开浏览器
    )
    print("Plot saved to sma_cross_result.html")
