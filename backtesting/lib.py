# ============================================================
# backtesting/lib.py — 策略辅助函数库
# ============================================================
# 上下文层：用户最常 import 的模块之一。提供信号判断、指标辅助、
#           可组合策略基类、多品种回测、数据生成器等工具。
# 功能层：crossover/cross 交叉判断、resample_apply 多时间框架、
#           SignalStrategy/TrailingStrategy 策略基类、
#           FractionalBacktest 分数股、MultiBacktest 多品种并行。
# 设计层：模板方法模式（SignalStrategy/TrailingStrategy 继承 Strategy）、
#          生成器（random_ohlc_data）、调用栈自省（resample_apply 检测 init 上下文）。

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
from collections import OrderedDict
from inspect import currentframe                 # 调用栈帧检测（resample_apply 用）
from itertools import chain, compress, count
from numbers import Number
from typing import Callable, Generator, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._plotting import plot_heatmaps as _plot_heatmaps
from ._stats import compute_stats as _compute_stats
from ._util import SharedMemoryManager, _Array, _as_str, _batch, _tqdm, patch
from .backtesting import Backtest, Strategy

__pdoc__ = {}


# ═══════════════════════════════════════════════════════════
# OHLCV 和交易数据重采样聚合规则
# ═══════════════════════════════════════════════════════════
# 上下文层：高频率 → 低频率重采样时，每列需要指定聚合方式。
# 设计层：OrderedDict 显式表达"列顺序重要"的意图。

OHLCV_AGG = OrderedDict((
    ('Open', 'first'),
    ('High', 'max'),
    ('Low', 'min'),
    ('Close', 'last'),
    ('Volume', 'sum'),
))
"""OHLCV 重采样聚合规则。用法：df.resample('4H', label='right').agg(OHLCV_AGG)"""

TRADES_AGG = OrderedDict((
    ('Size', 'sum'),
    ('EntryBar', 'first'),
    ('ExitBar', 'last'),
    ('EntryPrice', 'mean'),
    ('ExitPrice', 'mean'),
    ('PnL', 'sum'),
    ('ReturnPct', 'mean'),
    ('EntryTime', 'first'),
    ('ExitTime', 'last'),
    ('Duration', 'sum'),
))
"""交易数据重采样聚合规则。"""

_EQUITY_AGG = {
    'Equity': 'last',
    'DrawdownPct': 'max',
    'DrawdownDuration': 'max',
}


# ═══════════════════════════════════════════════════════════
# 信号判断工具
# ═══════════════════════════════════════════════════════════

def barssince(condition: Sequence[bool], default=np.inf) -> int:
    """
    返回离 condition 最近一次为 True 过了多少根 K 线。
    用 compress + reversed 实现——函数式风格。
    """
    return next(compress(range(len(condition)), reversed(condition)), default)


def cross(series1: Sequence, series2: Sequence) -> bool:
    """双向交叉判断——series1 上穿或下穿 series2 都算。"""
    return crossover(series1, series2) or crossover(series2, series1)


def crossover(series1: Sequence, series2: Sequence) -> bool:
    """
    上穿判断——run_demo.py 的 SmaCross 就用这个。
    输入可以是 ndarray、pd.Series 或常数。
    判断逻辑：前一 bar series1 < series2 且当前 bar series1 > series2。
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
        return series1[-2] < series2[-2] and series1[-1] > series2[-1]
    except IndexError:
        return False


def plot_heatmaps(heatmap: pd.Series,
                  agg: Union[str, Callable] = 'max',
                  *, ncols: int = 3, plot_width: int = 1200,
                  filename: str = '', open_browser: bool = True):
    """参数热力图——_plotting.plot_heatmaps 的代理，方便用户直接在 lib 层调用。"""
    return _plot_heatmaps(heatmap, agg, ncols, filename, plot_width, open_browser)


def quantile(series: Sequence, quantile: Union[None, float] = None):
    """
    分位数工具。
    quantile=None → 返回最新值在历史中的分位排名。
    quantile=0~1 → 返回序列在该分位的值。
    """
    if quantile is None:
        try:
            last, series = series[-1], series[:-1]
            return np.mean(series < last)
        except IndexError:
            return np.nan
    assert 0 <= quantile <= 1, "quantile must be within [0, 1]"
    return np.nanpercentile(series, quantile * 100)


# ═══════════════════════════════════════════════════════════
# 统计重算和重采样工具
# ═══════════════════════════════════════════════════════════

def compute_stats(*, stats: pd.Series, data: pd.DataFrame,
                  trades: pd.DataFrame = None,
                  risk_free_rate: float = 0.) -> pd.Series:
    """
    对部分交易（如仅多单）重新计算统计指标。
    如果只给 trades 子集，会重新构建权益曲线。
    """
    equity = stats._equity_curve.Equity
    if trades is None:
        trades = stats._trades
    else:
        equity = equity.copy()
        equity[:] = stats._equity_curve.Equity.iloc[0]
        for t in trades.itertuples(index=False):
            equity.iloc[t.EntryBar:] += t.PnL
    return _compute_stats(trades=trades, equity=equity.values,
                          ohlc_data=data, risk_free_rate=risk_free_rate,
                          strategy_instance=stats._strategy)


def resample_apply(rule: str,
                   func: Optional[Callable[..., Sequence]],
                   series: Union[pd.Series, pd.DataFrame, _Array],
                   *args, agg: Optional[Union[str, dict]] = None,
                   **kwargs):
    """
    多时间框架指标——把数据重采样到 rule 频率后计算 func。
    如果在 Strategy.init() 内调用，结果自动通过 self.I() 注册为指标。
    关键：用 inspect.currentframe() 遍历调用栈检测 init 上下文。
    """
    if func is None:
        def func(x, *_, **__):
            return x
    assert callable(func), 'resample_apply(func=) must be callable'

    if not isinstance(series, (pd.Series, pd.DataFrame)):
        assert isinstance(series, _Array), \
            'resample_apply(series=) must be pd.Series, pd.DataFrame, or Strategy.data.*'
        series = series.s

    if agg is None:
        agg = OHLCV_AGG.get(getattr(series, 'name', ''), 'last')
        if isinstance(series, pd.DataFrame):
            agg = {column: OHLCV_AGG.get(column, 'last')
                   for column in series.columns}

    # label='right' 是关键——用区间右端点避免 look-ahead bias
    resampled = series.resample(rule, label='right').agg(agg).dropna()
    resampled.name = _as_str(series) + '[' + rule + ']'

    # 调用栈检测——向上最多 3 帧找 Strategy.init 的 self.I
    frame, level = currentframe(), 0
    while frame and level <= 3:
        frame = frame.f_back
        level += 1
        if isinstance(frame.f_locals.get('self'), Strategy):
            strategy_I = frame.f_locals['self'].I
            break
    else:
        def strategy_I(func, *args, **kwargs):
            return func(*args, **kwargs)

    def wrap_func(resampled, *args, **kwargs):
        result = func(resampled, *args, **kwargs)
        if not isinstance(result, pd.DataFrame) and not isinstance(result, pd.Series):
            result = np.asarray(result)
            if result.ndim == 1:
                result = pd.Series(result, name=resampled.name)
            elif result.ndim == 2:
                result = pd.DataFrame(result.T)
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = resampled.index
        # ffill 向前填充——避免用未来数据
        result = result.reindex(index=series.index.union(resampled.index),
                                method='ffill').reindex(series.index)
        return result

    wrap_func.__name__ = func.__name__
    array = strategy_I(wrap_func, resampled, *args, **kwargs)
    return array


# ═══════════════════════════════════════════════════════════
# 随机数据生成器
# ═══════════════════════════════════════════════════════════

def random_ohlc_data(example_data: pd.DataFrame, *,
                     frac=1.,
                     random_state: Optional[int] = None
                     ) -> Generator[pd.DataFrame, None, None]:
    """
    无限生成随机 OHLC 数据，保持原始数据的统计分布。
    用于蒙特卡洛模拟和策略压力测试。
    Python 高级特性：生成器（while True + yield）。
    """
    def shuffle(x):
        return x.sample(frac=frac, replace=frac > 1,
                       random_state=random_state)

    if len(example_data.columns.intersection({'Open', 'High', 'Low', 'Close'})) != 4:
        raise ValueError("data must have columns Open, High, Low, Close")

    while True:
        df = shuffle(example_data)
        df.index = example_data.index
        padding = df.Close - df.Open.shift(-1)
        gaps = shuffle(example_data.Open.shift(-1) - example_data.Close)
        deltas = (padding + gaps).shift(1).fillna(0).cumsum()
        for key in ('Open', 'High', 'Low', 'Close'):
            df[key] += deltas
        yield df


# ═══════════════════════════════════════════════════════════
# 可组合策略基类
# ═══════════════════════════════════════════════════════════

class SignalStrategy(Strategy):
    """
    基于信号向量的策略——在 init() 里设置进场/出场信号数组，
    next() 自动处理订单。接近"矢量化回测"。
    使用方式：继承后调用 set_signal(entry, exit)。
    """
    __entry_signal = (0,)
    __exit_signal = (False,)

    def set_signal(self, entry_size: Sequence[float],
                   exit_portion: Optional[Sequence[float]] = None,
                   *, plot: bool = True):
        """设置进场/出场信号（散点图模式可视化）。"""
        self.__entry_signal = self.I(
            lambda: pd.Series(entry_size, dtype=float).replace(0, np.nan),
            name='entry size', plot=plot, overlay=False, scatter=True,
            color='black')

        if exit_portion is not None:
            self.__exit_signal = self.I(
                lambda: pd.Series(exit_portion, dtype=float).replace(0, np.nan),
                name='exit portion', plot=plot, overlay=False, scatter=True,
                color='black')

    def next(self):
        super().next()

        exit_portion = self.__exit_signal[-1]
        if exit_portion > 0:
            for trade in self.trades:
                if trade.is_long:
                    trade.close(exit_portion)
        elif exit_portion < 0:
            for trade in self.trades:
                if trade.is_short:
                    trade.close(-exit_portion)

        entry_size = self.__entry_signal[-1]
        if entry_size > 0:
            self.buy(size=entry_size)
        elif entry_size < 0:
            self.sell(size=-entry_size)


class TrailingStrategy(Strategy):
    """
    跟踪止损策略——以 ATR（平均真实波幅）的倍数为距离跟踪价格。
    多头止损价只升不降，空头止损价只降不升。
    """
    __n_atr = 6.
    __atr = None

    def init(self):
        super().init()
        self.set_atr_periods()

    def set_atr_periods(self, periods: int = 100):
        """设置 ATR 计算周期（默认 100 确保稳定）。"""
        hi, lo, c_prev = (self.data.High, self.data.Low,
                          pd.Series(self.data.Close).shift(1))
        tr = np.max([hi - lo, (c_prev - hi).abs(),
                    (c_prev - lo).abs()], axis=0)
        atr = pd.Series(tr).rolling(periods).mean().bfill().values
        self.__atr = atr

    def set_trailing_sl(self, n_atr: float = 6):
        """设置跟踪止损距离为 n_atr 倍 ATR。"""
        self.__n_atr = n_atr

    def set_trailing_pct(self, pct: float = .05):
        """设置跟踪止损距离为价格百分比（内部转成 ATR 倍数）。"""
        assert 0 < pct < 1, 'Need pct= as rate, i.e. 5% == 0.05'
        pct_in_atr = np.mean(self.data.Close * pct / self.__atr)
        self.set_trailing_sl(pct_in_atr)

    def next(self):
        super().next()
        index = len(self.data) - 1
        for trade in self.trades:
            if trade.is_long:
                trade.sl = max(trade.sl or -np.inf,
                               self.data.Close[index]
                               - self.__atr[index] * self.__n_atr)
            else:
                trade.sl = min(trade.sl or np.inf,
                               self.data.Close[index]
                               + self.__atr[index] * self.__n_atr)


# ═══════════════════════════════════════════════════════════
# 高级回测变体
# ═══════════════════════════════════════════════════════════

class FractionalBacktest(Backtest):
    """
    分数股回测——把价格×unit、成交量÷unit，用整股回测等价分数股。
    Python 高级特性：适配器模式。
    """
    def __init__(self, data, *args,
                 fractional_unit=1 / 100e6, **kwargs):
        if 'satoshi' in kwargs:
            warnings.warn(
                'Parameter FractionalBacktest(..., satoshi=) is deprecated. '
                'Use fractional_unit=.', DeprecationWarning, stacklevel=2)
            fractional_unit = 1 / kwargs.pop('satoshi')

        self._fractional_unit = fractional_unit
        self.__data: pd.DataFrame = data.copy(deep=False)
        for col in ('Open', 'High', 'Low', 'Close'):
            self.__data[col] *= self._fractional_unit
        for col in ('Volume',):
            self.__data[col] /= self._fractional_unit

        with warnings.catch_warnings(record=True):
            warnings.filterwarnings(action='ignore', message='frac')
            super().__init__(data, *args, **kwargs)

    def run(self, **kwargs) -> pd.Series:
        with patch(self, '_data', self.__data):
            result = super().run(**kwargs)

        trades: pd.DataFrame = result['_trades']
        trades['Size'] *= self._fractional_unit
        trades[['EntryPrice', 'ExitPrice', 'TP', 'SL']] /= self._fractional_unit

        indicators = result['_strategy']._indicators
        for indicator in indicators:
            if indicator._opts['overlay']:
                indicator /= self._fractional_unit

        return result


# 防止 pdoc3 文档化 Strategy 子类的 __init__（信息量低）
for cls in list(globals().values()):
    if isinstance(cls, type) and issubclass(cls, Strategy):
        __pdoc__[f'{cls.__name__}.__init__'] = False


class MultiBacktest:
    """
    多品种并行回测——同一策略在多个数据集上并行运行。
    用多进程 + 共享内存，避免 pickle 大数据集。
    """
    def __init__(self, df_list, strategy_cls, **kwargs):
        self._dfs = df_list
        self._strategy = strategy_cls
        self._bt_kwargs = kwargs

    def run(self, **kwargs):
        from . import Pool
        with Pool() as pool, SharedMemoryManager() as smm:
            shm = [smm.df2shm(df) for df in self._dfs]
            results = _tqdm(
                pool.imap(self._mp_task_run,
                          ((df_batch, self._strategy, self._bt_kwargs, kwargs)
                           for df_batch in _batch(shm))),
                total=len(shm), desc=self.run.__qualname__, mininterval=2)
            df = pd.DataFrame(list(chain(*results))).transpose()
        return df

    @staticmethod
    def _mp_task_run(args):
        data_shm, strategy, bt_kwargs, run_kwargs = args
        dfs, shms = zip(*(SharedMemoryManager.shm2df(i) for i in data_shm))
        try:
            return [stats.filter(regex='^[^_]') if stats['# Trades'] else None
                    for stats in (Backtest(df, strategy, **bt_kwargs)
                                  .run(**run_kwargs)
                                  for df in dfs)]
        finally:
            for shmem in chain(*shms):
                shmem.close()

    def optimize(self, **kwargs) -> pd.DataFrame:
        heatmaps = []
        for df in _tqdm(self._dfs, desc=self.__class__.__name__,
                       mininterval=2):
            bt = Backtest(df, self._strategy, **self._bt_kwargs)
            _best_stats, heatmap = bt.optimize(
                return_heatmap=True, return_optimization=False, **kwargs)
            heatmaps.append(heatmap)
        heatmap = pd.DataFrame(dict(zip(count(), heatmaps)))
        return heatmap


# ═══════════════════════════════════════════════════════════
# __all__ — 自动生成模块公开 API
# ═══════════════════════════════════════════════════════════
# 动态遍历全局符号：可调用且来自本模块，或全大写常量，且不以 _ 开头

__all__ = [getattr(v, '__name__', k)
           for k, v in globals().items()
           if ((callable(v)
                and getattr(v, '__module__', None) == __name__
                or k.isupper())
               and not getattr(v, '__name__', k).startswith('_'))]
