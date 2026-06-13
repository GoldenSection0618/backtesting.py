# ============================================================
# backtesting/lib.py — 策略辅助函数库和可组合策略基类
# ============================================================
# 上下文层：本模块是用户使用最频繁的模块之一。
#           提供了信号生成（cross/crossover）、指标辅助（resample_apply）、
#           可组合策略基类（SignalStrategy、TrailingStrategy）、
#           多品种回测（MultiBacktest）、分数股支持（FractionalBacktest）、
#           数据生成器（random_ohlc_data）和热力图（plot_heatmaps）等工具。
#           这些工具服务于用户自定义策略的快速开发和高级回测场景。
# ============================================================

# --- 模块顶层文档字符串 ---
"""
Collection of common building blocks, helper auxiliary functions and
composable strategy classes for reuse.

Intended for simple missing-link procedures, not reinventing
of better-suited, state-of-the-art, fast libraries,
such as TA-Lib, Tulipy, PyAlgoTrade, NumPy, SciPy ...

Please raise ideas for additions to this collection on the [issue tracker].

[issue tracker]: https://github.com/kernc/backtesting.py
"""

from __future__ import annotations

import warnings
from collections import OrderedDict               # 有序字典（OHLCV/Trades 聚合规则，保证列序）
from inspect import currentframe                  # 调用栈帧检查（resample_apply 检测 Strategy.init 上下文）
from itertools import chain, compress, count      # 迭代器工具（结果扁平化、条件压缩、热力图编号）
from numbers import Number                        # 数字类型检测
from typing import Callable, Generator, Optional, Sequence, Union

import numpy as np
import pandas as pd

# --- 内部模块导入 ---
from ._plotting import plot_heatmaps as _plot_heatmaps  # 热力图绘制（委托给 _plotting 模块）
from ._stats import compute_stats as _compute_stats     # 统计重算（委托给 _stats 模块）
from ._util import SharedMemoryManager, _Array, _as_str, _batch, _tqdm, patch
from .backtesting import Backtest, Strategy

# 功能层：pdoc3 文档生成器配置字典
# 设计层：通过设置 __pdoc__ 控制 pdoc3 的文档生成行为
__pdoc__ = {}


# ============================================================
# OHLCV_AGG —— OHLCV 数据重采样聚合规则
# ============================================================
# 上下文层：当用户需要将数据从高频率（如小时线）重采样到低频率（如日线）时，
#           需要定义每列的聚合方式。这个字典定义了金融数据标准的聚合规则。
# 设计层：使用 OrderedDict 保证列顺序（虽然 Python 3.7+ 普通 dict 也有序，
#          但显式使用 OrderedDict 表达了"顺序重要"的设计意图）。
OHLCV_AGG = OrderedDict((
    ('Open', 'first'),    # 开盘价：取区间第一个值
    ('High', 'max'),      # 最高价：取区间最大值
    ('Low', 'min'),       # 最低价：取区间最小值
    ('Close', 'last'),    # 收盘价：取区间最后一个值
    ('Volume', 'sum'),    # 成交量：取区间累计
))
"""Dictionary of rules for aggregating resampled OHLCV data frames,
e.g.

    df.resample('4H', label='right').agg(OHLCV_AGG).dropna()
"""

# ============================================================
# TRADES_AGG —— 交易数据重采样聚合规则
# ============================================================
# 上下文层：当统计模块需要对交易数据进行周期汇总时使用。
TRADES_AGG = OrderedDict((
    ('Size', 'sum'),         # 仓位大小：求和
    ('EntryBar', 'first'),   # 进场 K 线编号：取第一个
    ('ExitBar', 'last'),     # 出场 K 线编号：取最后一个
    ('EntryPrice', 'mean'),  # 进场价：取平均
    ('ExitPrice', 'mean'),   # 出场价：取平均
    ('PnL', 'sum'),          # 盈亏：求和
    ('ReturnPct', 'mean'),   # 收益率：取平均
    ('EntryTime', 'first'),  # 进场时间：取第一个
    ('ExitTime', 'last'),    # 出场时间：取最后一个
    ('Duration', 'sum'),     # 持仓时长：求和
))
"""Dictionary of rules for aggregating resampled trades data,
e.g.

    stats['_trades'].resample('1D', on='ExitTime',
                              label='right').agg(TRADES_AGG)
"""

# --- 权益曲线聚合规则 ---
# 功能层：重采样权益曲线时使用的聚合方式
_EQUITY_AGG = {
    'Equity': 'last',          # 权益：取区间最后值
    'DrawdownPct': 'max',      # 回撤百分比：取最大值
    'DrawdownDuration': 'max',  # 回撤持续期：取最大值
}


# ============================================================
# barssince —— 距条件最近一次为真的 K 线数
# ============================================================
# 设计层：使用 compress + reversed 的组合，这是一种简洁的函数式编程风格。
#          compress(data, selectors) 从 data 中取出 selectors 为 True 的元素。
#          将 condition 反转后，compress 返回的最后一个 True 的索引
#          即为"距最近一次为真"的偏移量。
def barssince(condition: Sequence[bool], default=np.inf) -> int:
    """
    Return the number of bars since `condition` sequence was last `True`,
    or if never, return `default`.

        >>> barssince(self.data.Close > self.data.Open)
        3
    """
    return next(compress(range(len(condition)), reversed(condition)), default)


# ============================================================
# cross —— 双向交叉判断
# ============================================================
# 功能层：判断两个序列是否刚交叉（无论方向）。
# 上下文层：在策略 next() 中用于检测交叉信号。
# 设计层：复用 crossover，先判断上穿再判断下穿。
def cross(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` and `series2` just crossed
    (above or below) each other.

        >>> cross(self.data.Close, self.sma)
        True

    """
    return crossover(series1, series2) or crossover(series2, series1)


# ============================================================
# crossover —— 上穿判断（最常用的信号函数）
# ============================================================
# 上下文层：这是回测策略中最常用的信号函数之一。
#           run_demo.py 中的 SmaCross 策略就使用它判断均线交叉。
# 设计层：处理三种输入类型——
#           pd.Series → 取 .values 作为 ndarray
#           数字 → 复用为常数序列
#           数组 → 直接使用
#          前一根的值 < 对方的前一根值，且当前值 > 对方当前值，即为"上穿"。
def crossover(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` just crossed over (above)
    `series2`.

        >>> crossover(self.data.Close, self.sma)
        True
    """
    series1 = (
        series1.values if isinstance(series1, pd.Series) else
        (series1, series1) if isinstance(series1, Number) else
        series1)
    series2 = (
        series2.values if isinstance(series2, pd.Series) else
        (series2, series2) if isinstance(series2, Number) else
        series2)
    try:
        return series1[-2] < series2[-2] and series1[-1] > series2[-1]  # type: ignore
    except IndexError:
        # 功能层：数据不足时返回 False（尚未交叉）
        return False


# ============================================================
# plot_heatmaps —— 参数热力图绘制
# ============================================================
# 上下文层：bt.optimize() 返回的参数扫描结果可以通过本函数可视化为热力图。
# 设计层：本函数是 _plotting.plot_heatmaps 的简单代理，
#          在这里提供只是为了更好的模块组织（用户只需 import lib 就能用）。
def plot_heatmaps(heatmap: pd.Series,
                  agg: Union[str, Callable] = 'max',
                  *,
                  ncols: int = 3,
                  plot_width: int = 1200,
                  filename: str = '',
                  open_browser: bool = True):
    """
    Plots a grid of heatmaps, one for every pair of parameters in `heatmap`.
    See example in [the tutorial].

    [the tutorial]: https://kernc.github.io/backtesting.py/doc/examples/Parameter%20Heatmap%20&%20Optimization.html#plot-heatmap  # noqa: E501

    `heatmap` is a Series as returned by
    `backtesting.backtesting.Backtest.optimize` when its parameter
    `return_heatmap=True`.

    When projecting the n-dimensional (n > 2) heatmap onto 2D, the values are
    aggregated by 'max' function by default. This can be tweaked
    with `agg` parameter, which accepts any argument pandas knows
    how to aggregate by.

    .. todo::
        Lay heatmaps out lower-triangular instead of in a simple grid.
        Like [`sambo.plot.plot_objective()`][plot_objective] does.

    [plot_objective]: \
        https://sambo-optimization.github.io/doc/sambo/plot.html#sambo.plot.plot_objective
    """
    return _plot_heatmaps(heatmap, agg, ncols, filename, plot_width, open_browser)


# ============================================================
# quantile —— 分位数工具
# ============================================================
# 功能层：支持两种模式——
#           1. quantile=None：返回最新值在历史中的分位排名
#           2. quantile=0~1：返回序列在指定分位的值
# 上下文层：在策略中用于判断当前价格/指标值是否处于"极端"位置。
def quantile(series: Sequence, quantile: Union[None, float] = None):
    """
    If `quantile` is `None`, return the quantile _rank_ of the last
    value of `series` wrt former series values.

    If `quantile` is a value between 0 and 1, return the _value_ of
    `series` at this quantile. If used to working with percentiles, just
    divide your percentile amount with 100 to obtain quantiles.

        >>> quantile(self.data.Close[-20:], .1)
        162.130
        >>> quantile(self.data.Close)
        0.13
    """
    if quantile is None:
        try:
            last, series = series[-1], series[:-1]
            return np.mean(series < last)   # 功能层：当前值在历史中的排名比
        except IndexError:
            return np.nan
    assert 0 <= quantile <= 1, "quantile must be within [0, 1]"
    return np.nanpercentile(series, quantile * 100)


# ============================================================
# compute_stats —— 统计指标重算（lib 层包装）
# ============================================================
# 上下文层：允许用户对部分交易（如仅多单）重新计算统计指标。
#           这是 post-hoc 分析的关键工具。
# 设计层：如果只提供 trades 子集，需要重新构建权益曲线——
#          从初始权益开始，逐步累加每笔交易的盈亏。
def compute_stats(
        *,
        stats: pd.Series,
        data: pd.DataFrame,
        trades: pd.DataFrame = None,
        risk_free_rate: float = 0.) -> pd.Series:
    """
    (Re-)compute strategy performance metrics.

    `stats` is the statistics series as returned by `backtesting.backtesting.Backtest.run()`.
    `data` is OHLC data as passed to the `backtesting.backtesting.Backtest`
    the `stats` were obtained in.
    `trades` can be a dataframe subset of `stats._trades` (e.g. only long trades).
    You can also tune `risk_free_rate`, used in calculation of Sharpe and Sortino ratios.

        >>> stats = Backtest(GOOG, MyStrategy).run()
        >>> only_long_trades = stats._trades[stats._trades.Size > 0]
        >>> long_stats = compute_stats(stats=stats, trades=only_long_trades,
        ...                            data=GOOG, risk_free_rate=.02)
    """
    equity = stats._equity_curve.Equity
    if trades is None:
        trades = stats._trades
    else:
        # 功能层：当只计算部分交易时，重新构建权益曲线
        equity = equity.copy()
        equity[:] = stats._equity_curve.Equity.iloc[0]  # 从初始权益开始
        for t in trades.itertuples(index=False):
            equity.iloc[t.EntryBar:] += t.PnL            # 逐笔累加盈亏
    return _compute_stats(trades=trades, equity=equity.values, ohlc_data=data,
                          risk_free_rate=risk_free_rate, strategy_instance=stats._strategy)


# ============================================================
# resample_apply —— 多时间框架指标计算
# ============================================================
# 上下文层：这是 lib.py 中最重要的函数之一。
#           允许用户在策略中对数据重采样（如从小时线→日线）后计算指标。
#           它自动检测调用上下文，如果在 Strategy.init() 中被调用，
#           则结果自动通过 Strategy.I() 注册为指标。
# 设计层：核心难点在于——
#           1. 通过 inspect.currentframe() 遍历调用栈来检测 Strategy.init 上下文
#           2. 重采样函数包装和数据索引对齐（前向填充）
#           3. 自动选择 OHLCV 列对应的聚合函数
def resample_apply(rule: str,
                   func: Optional[Callable[..., Sequence]],
                   series: Union[pd.Series, pd.DataFrame, _Array],
                   *args,
                   agg: Optional[Union[str, dict]] = None,
                   **kwargs):
    """
    Apply `func` (such as an indicator) to `series`, resampled to
    a time frame specified by `rule`. When called from inside
    `backtesting.backtesting.Strategy.init`,
    the result (returned) series will be automatically wrapped in
    `backtesting.backtesting.Strategy.I`
    wrapper method.

    `rule` is a valid [Pandas offset string] indicating
    a time frame to resample `series` to.

    [Pandas offset string]: \
http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    `func` is the indicator function to apply on the resampled series.

    `series` is a data series (or array), such as any of the
    `backtesting.backtesting.Strategy.data` series. Due to pandas
    resampling limitations, this only works when input series
    has a datetime index.

    `agg` is the aggregation function to use on resampled groups of data.
    Valid values are anything accepted by `pandas/resample/.agg()`.
    Default value for dataframe input is `OHLCV_AGG` dictionary.
    Default value for series input is the appropriate entry from `OHLCV_AGG`
    if series has a matching name, or otherwise the value `"last"`,
    which is suitable for closing prices,
    but you might prefer another (e.g. `"max"` for peaks, or similar).

    Finally, any `*args` and `**kwargs` that are not already eaten by
    implicit `backtesting.backtesting.Strategy.I` call
    are passed to `func`.

    For example, if we have a typical moving average function
    `SMA(values, lookback_period)`, _hourly_ data source, and need to
    apply the moving average MA(10) on a _daily_ time frame,
    but don't want to plot the resulting indicator, we can do:

        class System(Strategy):
            def init(self):
                self.sma = resample_apply(
                    'D', SMA, self.data.Close, 10, plot=False)

    The above short snippet is roughly equivalent to:

        class System(Strategy):
            def init(self):
                # Strategy exposes `self.data` as raw NumPy arrays.
                # Let's convert closing prices back to pandas Series.
                close = self.data.Close.s

                # Resample to daily resolution. Aggregate groups
                # using their last value (i.e. closing price at the end
                # of the day). Notice `label='right'`. If it were set to
                # 'left' (default), the strategy would exhibit
                # look-ahead bias.
                daily = close.resample('D', label='right').agg('last')

                # We apply SMA(10) to daily close prices,
                # then reindex it back to original hourly index,
                # forward-filling the missing values in each day.
                # We make a separate function that returns the final
                # indicator array.
                def SMA(series, n):
                    from backtesting.test import SMA
                    return SMA(series, n).reindex(close.index).ffill()

                # The result equivalent to the short example above:
                self.sma = self.I(SMA, daily, 10, plot=False)

    """
    # 功能层：当 func 为 None 时，返回原始数据（仅做重采样，不计算指标）
    if func is None:
        def func(x, *_, **__):
            return x
    assert callable(func), 'resample_apply(func=) must be callable'

    # 功能层：将 _Array 类型转换为 pandas Series（重采样需要 DatetimeIndex）
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        assert isinstance(series, _Array), \
            'resample_apply(series=) must be `pd.Series`, `pd.DataFrame`, ' \
            'or a `Strategy.data.*` array'
        series = series.s

    # --- 自动选择聚合函数 ---
    if agg is None:
        # 功能层：根据列名从 OHLCV_AGG 中匹配；未匹配则默认 'last'
        agg = OHLCV_AGG.get(getattr(series, 'name', ''), 'last')
        if isinstance(series, pd.DataFrame):
            # 功能层：DataFrame 时对每列分别选择聚合函数
            agg = {column: OHLCV_AGG.get(column, 'last')
                   for column in series.columns}

    # 功能层：执行重采样。label='right' 至关重要——
    #          使用区间右端点（而非左端点）作为标签，
    #          避免使用区间终点之后的数据（look-ahead bias）
    resampled = series.resample(rule, label='right').agg(agg).dropna()
    resampled.name = _as_str(series) + '[' + rule + ']'

    # --- 调用栈检测：判断是否在 Strategy.init() 中调用 ---
    # 设计层：通过 inspect.currentframe() 获取当前执行帧，
    #          然后向上遍历调用栈（最多 3 层），
    #          查找调用者的 self 是否为 Strategy 实例。
    #          如果找到，则使用策略的 self.I() 方法自动注册指标。
    #          Python 高级特性：调用栈自省（frame introspection）。
    frame, level = currentframe(), 0
    while frame and level <= 3:
        frame = frame.f_back  # 向上一帧
        level += 1
        if isinstance(frame.f_locals.get('self'), Strategy):  # type: ignore
            strategy_I = frame.f_locals['self'].I             # type: ignore
            break
    else:
        # 功能层：非 Strategy.init 调用时，提供一个透传的函数包装
        def strategy_I(func, *args, **kwargs):  # noqa: F811
            return func(*args, **kwargs)

    # --- 指标计算包装函数 ---
    def wrap_func(resampled, *args, **kwargs):
        result = func(resampled, *args, **kwargs)
        # 功能层：将结果转换为 pandas Series/DataFrame
        if not isinstance(result, pd.DataFrame) and not isinstance(result, pd.Series):
            result = np.asarray(result)
            if result.ndim == 1:
                result = pd.Series(result, name=resampled.name)
            elif result.ndim == 2:
                result = pd.DataFrame(result.T)
        # 功能层：将重采样后的结果映射回原始时间索引
        #         使用 ffill（前向填充）避免使用未来数据
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = resampled.index
        result = result.reindex(index=series.index.union(resampled.index),
                                method='ffill').reindex(series.index)
        return result

    # 功能层：保留原始函数名（用于图表图例）
    wrap_func.__name__ = func.__name__

    # 功能层：通过 I() 注册指标，使其参与图表绘制和统计
    array = strategy_I(wrap_func, resampled, *args, **kwargs)
    return array


# ============================================================
# random_ohlc_data —— 随机OHLC数据生成器
# ============================================================
# 上下文层：用于策略的蒙特卡洛模拟和压力测试。
#           生成的随机数据保持了原始数据的统计特征（通过重采样+偏移）。
# 设计层：这是一个无限生成器（while True + yield），
#          每次迭代生成一组新的随机数据。
#          Python 高级特性：生成器（Generator）。
def random_ohlc_data(example_data: pd.DataFrame, *,
                     frac=1., random_state: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
    """
    OHLC data generator. The generated OHLC data has basic
    [descriptive statistics](https://en.wikipedia.org/wiki/Descriptive_statistics)
    similar to the provided `example_data`.

    `frac` is a fraction of data to sample (with replacement). Values greater
    than 1 result in oversampling.

    Such random data can be effectively used for stress testing trading
    strategy robustness, Monte Carlo simulations, significance testing, etc.

    >>> from backtesting.test import EURUSD
    >>> ohlc_generator = random_ohlc_data(EURUSD)
    >>> next(ohlc_generator)  # returns new random data
    ...
    >>> next(ohlc_generator)  # returns new random data
    ...
    """
    # 功能层：对数据行进行有放回抽样（保持统计分布）
    def shuffle(x):
        return x.sample(frac=frac, replace=frac > 1, random_state=random_state)

    # 功能层：验证输入包含必需的 OHLC 列
    if len(example_data.columns.intersection({'Open', 'High', 'Low', 'Close'})) != 4:
        raise ValueError("`data` must be a pandas.DataFrame with columns "
                         "'Open', 'High', 'Low', 'Close'")

    # 功能层：无限生成循环
    while True:
        df = shuffle(example_data)              # 随机抽样行
        df.index = example_data.index           # 恢复原始时间索引
        # 功能层：构建价格偏移量以保持价格序列的连续性
        padding = df.Close - df.Open.shift(-1)  # 本根收盘与下根开盘的价差
        gaps = shuffle(example_data.Open.shift(-1) - example_data.Close)  # 随机跳空
        deltas = (padding + gaps).shift(1).fillna(0).cumsum()  # 累积偏移
        for key in ('Open', 'High', 'Low', 'Close'):
            df[key] += deltas  # 将偏移应用到所有价格列
        yield df


# ============================================================
# SignalStrategy —— 基于信号向量的策略
# ============================================================
# 上下文层：将回测从"逐 K 线判断"简化为"设置信号向量"。
#           用户只需在 init() 中设置进场/出场信号数组，
#           next() 中框架自动处理订单。
#           这使得回测接近"矢量化回测"（vectorized backtest）。
# 设计层：使用 I() 注册信号为散点指标，以便在图表中可视化。
#          Python 高级特性：模板方法模式（用户覆盖 init/set_signal，框架调用 next）。
class SignalStrategy(Strategy):
    """
    A simple helper strategy that operates on position entry/exit signals.
    This makes the backtest of the strategy simulate a [vectorized backtest].
    See [tutorials] for usage examples.

    [vectorized backtest]: https://www.google.com/search?q=vectorized+backtest
    [tutorials]: index.html#tutorials

    To use this helper strategy, subclass it, override its
    `backtesting.backtesting.Strategy.init` method,
    and set the signal vector by calling
    `backtesting.lib.SignalStrategy.set_signal` method from within it.

        class ExampleStrategy(SignalStrategy):
            def init(self):
                super().init()
                self.set_signal(sma1 > sma2, sma1 < sma2)

    Remember to call `super().init()` and `super().next()` in your
    overridden methods.
    """
    # 功能层：默认的进场/出场信号（空信号）
    __entry_signal = (0,)
    __exit_signal = (False,)

    def set_signal(self, entry_size: Sequence[float],
                   exit_portion: Optional[Sequence[float]] = None,
                   *,
                   plot: bool = True):
        """
        Set entry/exit signal vectors (arrays).

        A long entry signal is considered present wherever `entry_size`
        is greater than zero, and a short signal wherever `entry_size`
        is less than zero, following `backtesting.backtesting.Order.size` semantics.

        If `exit_portion` is provided, a nonzero value closes portion the position
        (see `backtesting.backtesting.Trade.close()`) in the respective direction
        (positive values close long trades, negative short).

        If `plot` is `True`, the signal entry/exit indicators are plotted when
        `backtesting.backtesting.Backtest.plot` is called.
        """
        # 功能层：进场信号——使用散点图模式（scatter=True），NaN 替换为 0 在图表中不显示
        self.__entry_signal = self.I(  # type: ignore
            lambda: pd.Series(entry_size, dtype=float).replace(0, np.nan),
            name='entry size', plot=plot, overlay=False, scatter=True, color='black')

        if exit_portion is not None:
            # 功能层：出场信号——与进场信号相同的散点图显示方式
            self.__exit_signal = self.I(  # type: ignore
                lambda: pd.Series(exit_portion, dtype=float).replace(0, np.nan),
                name='exit portion', plot=plot, overlay=False, scatter=True, color='black')

    def next(self):
        super().next()

        # --- 处理出场信号 ---
        exit_portion = self.__exit_signal[-1]
        if exit_portion > 0:
            # 功能层：正值 = 关闭部分多头仓位
            for trade in self.trades:
                if trade.is_long:
                    trade.close(exit_portion)
        elif exit_portion < 0:
            # 功能层：负值 = 关闭部分空头仓位
            for trade in self.trades:
                if trade.is_short:
                    trade.close(-exit_portion)

        # --- 处理进场信号 ---
        entry_size = self.__entry_signal[-1]
        if entry_size > 0:
            self.buy(size=entry_size)    # 功能层：正值 = 多头进场
        elif entry_size < 0:
            self.sell(size=-entry_size)  # 功能层：负值 = 空头进场


# ============================================================
# TrailingStrategy —— 跟踪止损策略
# ============================================================
# 上下文层：提供自动跟踪止损功能。止损价以 ATR（平均真实波幅）
#           的倍数为距离跟踪当前价格。这是趋势跟踪策略中常用的风险管理工具。
# 设计层：ATR 计算使用 pandas rolling + bfill（后向填充），
#          确保在数据开始时就有一个合理的 ATR 值。
#          Python 高级特性：模板方法模式 + 策略模式。
class TrailingStrategy(Strategy):
    """
    A strategy with automatic trailing stop-loss, trailing the current
    price at distance of some multiple of average true range (ATR). Call
    `TrailingStrategy.set_trailing_sl()` to set said multiple
    (`6` by default). See [tutorials] for usage examples.

    [tutorials]: index.html#tutorials

    Remember to call `super().init()` and `super().next()` in your
    overridden methods.
    """
    # 功能层：默认 6 倍 ATR 作为跟踪止损距离
    __n_atr = 6.
    __atr = None

    def init(self):
        super().init()
        # 功能层：初始化时自动计算 ATR（默认 100 周期）
        self.set_atr_periods()

    def set_atr_periods(self, periods: int = 100):
        """
        Set the lookback period for computing ATR. The default value
        of 100 ensures a _stable_ ATR.
        """
        # 功能层：计算 True Range（真实波幅）
        #          TR = max(High-Low, |Close_prev-High|, |Close_prev-Low|)
        hi, lo, c_prev = self.data.High, self.data.Low, pd.Series(self.data.Close).shift(1)
        tr = np.max([hi - lo, (c_prev - hi).abs(), (c_prev - lo).abs()], axis=0)
        # 功能层：TR 的滚动均值就是 ATR
        atr = pd.Series(tr).rolling(periods).mean().bfill().values
        self.__atr = atr

    def set_trailing_sl(self, n_atr: float = 6):
        """
        Set the future trailing stop-loss as some multiple (`n_atr`)
        average true bar ranges away from the current price.
        """
        self.__n_atr = n_atr

    def set_trailing_pct(self, pct: float = .05):
        """
        Set the future trailing stop-loss as some percent (`0 < pct < 1`)
        below the current price (default 5% below).

        .. note:: Stop-loss set by `pct` is inexact
            Stop-loss set by `set_trailing_pct` is converted to units of ATR
            with `mean(Close * pct / atr)` and set with `set_trailing_sl`.
        """
        assert 0 < pct < 1, 'Need pct= as rate, i.e. 5% == 0.05'
        # 功能层：将百分比止损转换为 ATR 倍数止损
        pct_in_atr = np.mean(self.data.Close * pct / self.__atr)  # type: ignore
        self.set_trailing_sl(pct_in_atr)

    def next(self):
        super().next()
        # 上下文层：__atr 不是 Indicator 类型，不能用 index=-1，需要手动计算当前索引
        index = len(self.data) - 1
        for trade in self.trades:
            if trade.is_long:
                # 功能层：多头止损价 = max(原止损价, 当前价 - n*ATR)
                #          止损价只能向上移动（跟踪），不能向下移动
                trade.sl = max(trade.sl or -np.inf,
                               self.data.Close[index] - self.__atr[index] * self.__n_atr)
            else:
                # 功能层：空头止损价 = min(原止损价, 当前价 + n*ATR)
                #          止损价只能向下移动（跟踪），不能向上移动
                trade.sl = min(trade.sl or np.inf,
                               self.data.Close[index] + self.__atr[index] * self.__n_atr)


# ============================================================
# FractionalBacktest —— 分数股回测
# ============================================================
# 上下文层：Backtesting.py 默认使用整数股交易（整股）。
#           本类通过对价格和成交量的缩放变换，
#           使得"整股回测"等价于"分数股回测"。
# 设计层：核心技巧——将价格乘以 fractional_unit，
#          成交量除以 fractional_unit，
#          回测完成后反向还原指标和交易数据。
#          Python 高级特性：适配器模式。
class FractionalBacktest(Backtest):
    """
    A `backtesting.backtesting.Backtest` that supports fractional share trading
    by simple composition. It applies roughly the transformation:

        data = (data * fractional_unit).assign(Volume=data.Volume / fractional_unit)

    as left unchallenged in [this FAQ entry on GitHub](https://github.com/kernc/backtesting.py/issues/134),
    then passes `data`, `args*`, and `**kwargs` to its super.

    Parameter `fractional_unit` represents the smallest fraction of currency that can be traded
    and defaults to one [satoshi]. For μBTC trading, pass `fractional_unit=1/1e6`.
    Thus-transformed backtest does a whole-sized trading of `fractional_unit` units.

    [satoshi]: https://en.wikipedia.org/wiki/Bitcoin#Units_and_divisibility
    """
    def __init__(self,
                 data,
                 *args,
                 fractional_unit=1 / 100e6,
                 **kwargs):
        # 功能层：支持已弃用的 satoshi= 参数（向后兼容）
        if 'satoshi' in kwargs:
            warnings.warn(
                'Parameter `FractionalBacktest(..., satoshi=)` is deprecated. '
                'Use `FractionalBacktest(..., fractional_unit=)`.',
                category=DeprecationWarning, stacklevel=2)
            fractional_unit = 1 / kwargs.pop('satoshi')

        self._fractional_unit = fractional_unit
        # 功能层：浅拷贝原始数据（deep=False），避免不必要的大内存拷贝
        self.__data: pd.DataFrame = data.copy(deep=False)
        # 功能层：缩放变换——价格 *= unit, 成交量 /= unit
        for col in ('Open', 'High', 'Low', 'Close',):
            self.__data[col] = self.__data[col] * self._fractional_unit
        for col in ('Volume',):
            self.__data[col] = self.__data[col] / self._fractional_unit

        # 功能层：抑制有关 frac 的警告
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings(action='ignore', message='frac')
            super().__init__(data, *args, **kwargs)

    def run(self, **kwargs) -> pd.Series:
        # 功能层：用变换后的数据临时替换原始数据
        with patch(self, '_data', self.__data):
            result = super().run(**kwargs)

        # 功能层：还原交易结果——
        #         仓位放回原始尺度，价格除以缩放因子
        trades: pd.DataFrame = result['_trades']
        trades['Size'] *= self._fractional_unit
        trades[['EntryPrice', 'ExitPrice', 'TP', 'SL']] /= self._fractional_unit

        # 功能层：还原 overlay 类指标（价格相关指标需要反向缩放）
        indicators = result['_strategy']._indicators
        for indicator in indicators:
            if indicator._opts['overlay']:
                indicator /= self._fractional_unit

        return result


# --- 防止 pdoc3 文档化 Strategy 子类的 __init__ 签名 ---
# 功能层：pdoc 会自动文档化继承的 __init__，但这对于策略子类来说意义不大。
#         这里遍历所有全局变量，对 Strategy 的子类禁用 __init__ 的文档生成。
for cls in list(globals().values()):
    if isinstance(cls, type) and issubclass(cls, Strategy):
        __pdoc__[f'{cls.__name__}.__init__'] = False


# ============================================================
# MultiBacktest —— 多品种并行回测
# ============================================================
# 上下文层：允许同一策略在多个数据集（如多只股票/多币种）上并行运行。
#           用于跨市场/跨品种策略对比和优化。
# 设计层：使用多进程池（backtesting.Pool）并行执行各个品种的回测。
#          数据和结果通过共享内存（SharedMemoryManager）传递，
#          避免 pickle 大数据集到每个子进程。
#          Python 高级特性：多进程 + 共享内存 + 上下文管理器协议。
class MultiBacktest:
    """
    Multi-dataset `backtesting.backtesting.Backtest` wrapper.

    Run supplied `backtesting.backtesting.Strategy` on several instruments,
    in parallel.  Used for comparing strategy runs across many instruments
    or classes of instruments. Example:

        from backtesting.test import EURUSD, BTCUSD, SmaCross
        btm = MultiBacktest([EURUSD, BTCUSD], SmaCross)
        stats_per_ticker: pd.DataFrame = btm.run(fast=10, slow=20)
        heatmap_per_ticker: pd.DataFrame = btm.optimize(...)
    """
    def __init__(self, df_list, strategy_cls, **kwargs):
        self._dfs = df_list             # 多个品种的 OHLC 数据
        self._strategy = strategy_cls   # 策略类（非实例）
        self._bt_kwargs = kwargs        # Backtest 构造参数（commission 等）

    def run(self, **kwargs):
        """
        Wraps `backtesting.backtesting.Backtest.run`. Returns `pd.DataFrame` with
        currency indexes in columns.
        """
        from . import Pool  # 延迟导入，使用用户可替换的 Pool
        # 设计层：使用两个上下文管理器——
        #          Pool() 管理多进程池，
        #          SharedMemoryManager() 管理共享内存生命周期
        with Pool() as pool, \
                SharedMemoryManager() as smm:
            # 功能层：将每个品种的 DataFrame 序列化到共享内存
            shm = [smm.df2shm(df) for df in self._dfs]
            # 功能层：使用 imap 并行处理每个品种
            results = _tqdm(
                pool.imap(self._mp_task_run,
                          ((df_batch, self._strategy, self._bt_kwargs, kwargs)
                           for df_batch in _batch(shm))),
                total=len(shm),
                desc=self.run.__qualname__,
                mininterval=2
            )
            # 功能层：汇总结果为 DataFrame（行=指标，列=品种）
            df = pd.DataFrame(list(chain(*results))).transpose()
        return df

    # 设计层：静态方法，可被 pickle 序列化用于多进程序列化
    @staticmethod
    def _mp_task_run(args):
        """每个子进程中的回测执行任务"""
        data_shm, strategy, bt_kwargs, run_kwargs = args
        # 功能层：从共享内存恢复 DataFrame 数据
        dfs, shms = zip(*(SharedMemoryManager.shm2df(i) for i in data_shm))
        try:
            # 功能层：过滤掉内部字段（以 _ 开头），只返回公开的统计指标
            return [stats.filter(regex='^[^_]') if stats['# Trades'] else None
                    for stats in (Backtest(df, strategy, **bt_kwargs).run(**run_kwargs)
                                  for df in dfs)]
        finally:
            # 功能层：确保退出时关闭所有共享内存连接
            for shmem in chain(*shms):
                shmem.close()

    def optimize(self, **kwargs) -> pd.DataFrame:
        """
        Wraps `backtesting.backtesting.Backtest.optimize`, but returns `pd.DataFrame` with
        currency indexes in columns.

            heamap: pd.DataFrame = btm.optimize(...)
            from backtesting.plot import plot_heatmaps
            plot_heatmaps(heatmap.mean(axis=1))
        """
        heatmaps = []
        # 功能层：bt.optimize() 自己内部使用多进程，
        #         所以这里只做简单循环（避免嵌套多进程）
        for df in _tqdm(self._dfs, desc=self.__class__.__name__, mininterval=2):
            bt = Backtest(df, self._strategy, **self._bt_kwargs)
            _best_stats, heatmap = bt.optimize(  # type: ignore
                return_heatmap=True, return_optimization=False, **kwargs)
            heatmaps.append(heatmap)
        # 功能层：汇总所有品种的热力图为 DataFrame
        heatmap = pd.DataFrame(dict(zip(count(), heatmaps)))
        return heatmap


# ============================================================
# __all__ —— 模块公开 API 列表
# ============================================================
# 上下文层：控制 `from backtesting.lib import *` 的行为。
#           只有在 __all__ 中列出的名称才会被导出。
# 设计层：自动生成——遍历所有全局符号，
#          条件：可调用且来自本模块，或全大写的常量，且不以 '_' 开头。
#          这种动态方式比手动维护 __all__ 更健壮。
# NOTE: Don't put anything below this __all__ list

__all__ = [getattr(v, '__name__', k)
           for k, v in globals().items()                        # export
           if ((callable(v) and getattr(v, '__module__', None) == __name__ or  # callables from this module
                k.isupper()) and                                # or CONSTANTS
               not getattr(v, '__name__', k).startswith('_'))]  # neither marked internal

# NOTE: Don't put anything below here. See above.
