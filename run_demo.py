# ============================================================
# run_demo.py —— Backtesting.py 最小运行示例
# ============================================================
# 上下文层：本脚本复现经典的 SMA（简单移动平均线）交叉策略。
#           是课程大作业的运行验证入口，证明精简后的工程可正常回测。
# 功能层：使用 GOOG 日线数据，SMA(10) 与 SMA(20) 的交叉产生交易信号，
#          输出回测统计结果并生成交互式 HTML 图表。
# 设计层：SmaCross 继承 Strategy 基类，实现 init()（预计算指标）
#          和 next()（交易决策），体现了模板方法模式。
# ============================================================

# --- 框架核心导入 ---
# Backtest：回测引擎，Strategy：策略基类（用户必须继承）
from backtesting import Backtest, Strategy
# crossover：上穿判断函数（series1 从下方突破 series2）
from backtesting.lib import crossover
# GOOG：Google 历史日线数据（2004-2013），SMA：简单移动平均线计算函数
from backtesting.test import GOOG, SMA


# ============================================================
# SmaCross —— SMA 交叉策略
# ============================================================
# 上下文层：这是 Backtesting.py 官方最简示例策略。
#           SMA(10) 上穿 SMA(20) 时做多，下穿时做空。
# 设计层：类变量 n1/n2 是可优化参数，Backtest.optimize() 可以网格搜索最佳值。
class SmaCross(Strategy):
    # 功能层：类变量定义策略参数（同时作为 optimize 的参数搜索范围）
    n1 = 10   # 快线周期
    n2 = 20   # 慢线周期

    def init(self):
        # 功能层：策略初始化——在回测开始前一次性预计算全部指标
        #          这里使用的是"全量数据"（不像 next() 中数据逐步揭示）
        price = self.data.Close                        # 收盘价数组（_Array 类型，ndarray 子类）
        self.ma1 = self.I(SMA, price, self.n1)         # 注册快线指标（self.I 是 Strategy.I）
        self.ma2 = self.I(SMA, price, self.n2)         # 注册慢线指标

    def next(self):
        # 功能层：每根新 K 线到来时调用一次（逐 K 线推进）
        #          当前数据长度 = i+1（i 为当前 K 线编号）
        #          self.ma1[-1] 表示快线的最新值，self.ma1[-2] 为上一根值
        if crossover(self.ma1, self.ma2):
            # 功能层：快线上穿慢线 → 做多信号
            self.buy()
        elif crossover(self.ma2, self.ma1):
            # 功能层：快线下穿慢线 → 做空信号
            self.sell()


# --- 主程序入口 ---
if __name__ == "__main__":
    # 功能层：创建回测实例
    #   - GOOG：OHLCV 行情数据（pd.DataFrame）
    #   - SmaCross：策略类（非实例，Backtest 内部会实例化）
    #   - commission=0.002：佣金率 0.2%（买卖各收一次）
    #   - exclusive_orders=True：新订单自动平掉旧仓位（同一时间最多一个仓位）
    bt = Backtest(
        GOOG,
        SmaCross,
        commission=0.002,
        exclusive_orders=True,
    )

    # 功能层：执行回测，返回 pd.Series 统计结果（约 30 项指标）
    stats = bt.run()
    print(stats)  # 打印回测统计（Start/End/Return/Sharpe/Max Drawdown 等）

    # 功能层：生成交互式 HTML 图表
    #   - filename：输出文件名
    #   - open_browser=False：不自动打开浏览器（适合无 GUI 的服务器环境）
    bt.plot(
        filename="sma_cross_result.html",
        open_browser=False,
    )
    print("Plot saved to sma_cross_result.html")
