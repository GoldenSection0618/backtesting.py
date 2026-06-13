# ============================================================
# backtesting/backtesting.py — 核心回测引擎
# ============================================================
# 上下文层：这是整个 Backtesting.py 框架的核心。
#           包含回测引擎 Backtest、策略基类 Strategy、
#           订单 Order、交易 Trade、仓位 Position、内部 Broker 等全部核心类。
#           整个框架的能力——事件驱动逐K线回测、策略注册、订单管理、
#           交易撮合、参数优化、图表绘制——都由本模块驱动。
# 设计层：面向对象设计，使用继承+模板方法模式。
#           用户继承 Strategy 实现 init()/next()，
#           Backtest.run() 驱动主循环，_Broker 管理订单/交易状态。
#           Python 高级特性：元类（ABCMeta）、抽象方法、属性（property）、
#           静态方法、partial 函数、pickle 序列化、多进程优化。
# ============================================================

# --- 模块顶层文档字符串 ---
"""
Core framework data structures.
Objects from this module can also be imported from the top-level
module directly, e.g.

    from backtesting import Backtest, Strategy
"""

from __future__ import annotations

import sys
import warnings
from abc import ABCMeta, abstractmethod          # 抽象基类元类和装饰器（Strategy 定义接口）
from copy import copy                             # 浅拷贝（Trade._copy 创建部分平仓副本）
from difflib import get_close_matches             # 相似字符串匹配（参数名错误时的友好提示）
from functools import lru_cache, partial          # LRU缓存（优化去重）和偏函数（Broker 延迟构造）
from itertools import chain, product, repeat      # 迭代器工具（参数排列、索引扁平化）
from math import copysign                         # 符号复制（处理多头/空头的相反方向）
from numbers import Number                        # 数字类型检测
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng             # NumPy 默认随机数生成器（优化网格搜索）

# --- 内部模块导入 ---
from ._plotting import plot  # noqa: I001         # 图表绘制（Backtest.plot 委托）
from ._stats import compute_stats, dummy_stats    # 统计计算和字段探测
from ._util import (
    SharedMemoryManager, _as_str, _Indicator, _Data, _batch, _indicator_warmup_nbars,
    _strategy_indicators, patch, try_, _tqdm,
)

# 功能层：pdoc3 文档生成配置——隐藏内部类的 __init__ 文档
__pdoc__ = {
    'Strategy.__init__': False,
    'Order.__init__': False,
    'Position.__init__': False,
    'Trade.__init__': False,
}


# ============================================================
# Strategy —— 策略基类
# ============================================================
# 上下文层：这是用户必须继承的核心类。
#           用户实现 init()（预计算指标）和 next()（交易决策），
#           框架在回测中自动调用它们。
# 设计层：使用 ABCMeta 元类 + @abstractmethod 强制子类实现接口。
#          这是"模板方法"模式——框架定义骨架，用户填充细节。
#          Python 高级特性：元类、抽象基类、property。
class Strategy(metaclass=ABCMeta):
    """
    A trading strategy base class. Extend this class and
    override methods
    `backtesting.backtesting.Strategy.init` and
    `backtesting.backtesting.Strategy.next` to define
    your own strategy.
    """
    def __init__(self, broker, data, params):
        self._indicators = []         # 已注册的指标列表（_Indicator 实例）
        self._broker: _Broker = broker  # 内部 Broker 引用（订单/交易/权益管理）
        self._data: _Data = data        # OHLCV 数据访问器（_Data 封装）
        self._params = self._check_params(params)  # 参数校验与注入

    def __repr__(self):
        return '<Strategy ' + str(self) + '>'

    def __str__(self):
        # 功能层：生成类似 "SmaCross(n1=10,n2=20)" 的字符串表示
        params = ','.join(f'{i[0]}={i[1]}' for i in zip(self._params.keys(),
                                                        map(_as_str, self._params.values())))
        if params:
            params = '(' + params + ')'
        return f'{self.__class__.__name__}{params}'

    # --- 参数校验 ---
    # 功能层：检查传入的参数是否在类中有对应的类变量定义。
    # 设计层：使用 get_close_matches 提供"你是不是想写...?"的友好错误提示。
    #          这是一种防御性编程实践。
    def _check_params(self, params):
        for k, v in params.items():
            if not hasattr(self, k):
                suggestions = get_close_matches(k, (attr for attr in dir(self) if not attr.startswith('_')))
                hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise AttributeError(
                    f"Strategy '{self.__class__.__name__}' is missing parameter '{k}'. "
                    "Strategy class should define parameters as class variables before they "
                    "can be optimized or run with." + hint)
            setattr(self, k, v)        # 功能层：动态设置策略实例的参数值
        return params

    # ====== Strategy.I —— 指标注册 ======
    # 上下文层：这是策略中最关键的方法之一。
    #           用户在 init() 中调用 self.I() 来声明技术指标（如 SMA、RSI）。
    #           框架负责：1. 在回测中逐步揭示指标值（避免未来信息）
    #                     2. 在图表上绘制指标
    # 设计层：I 这个名字故意简短，因为它是用户最频繁调用的 API。
    def I(self,  # noqa: E743
          func: Callable, *args,
          name=None, plot=True, overlay=None, color=None, scatter=False,
          **kwargs) -> np.ndarray:
        """
        Declare an indicator. An indicator is just an array of values
        (or a tuple of such arrays in case of, e.g., MACD indicator),
        but one that is revealed gradually in
        `backtesting.backtesting.Strategy.next` much like
        `backtesting.backtesting.Strategy.data` is.
        Returns `np.ndarray` of indicator values.

        `func` is a function that returns the indicator array(s) of
        same length as `backtesting.backtesting.Strategy.data`.

        In the plot legend, the indicator is labeled with
        function name, unless `name` overrides it. If `func` returns
        a tuple of arrays, `name` can be a sequence of strings, and
        its size must agree with the number of arrays returned.

        If `plot` is `True`, the indicator is plotted on the resulting
        `backtesting.backtesting.Backtest.plot`.

        If `overlay` is `True`, the indicator is plotted overlaying the
        price candlestick chart (suitable e.g. for moving averages).
        If `False`, the indicator is plotted standalone below the
        candlestick chart. By default, a heuristic is used which decides
        correctly most of the time.

        `color` can be string hex RGB triplet or X11 color name.
        By default, the next available color is assigned.

        If `scatter` is `True`, the plotted indicator marker will be a
        circle instead of a connected line segment (default).

        Additional `*args` and `**kwargs` are passed to `func` and can
        be used for parameters.

        For example, using simple moving average function from TA-Lib:

            def init():
                self.sma = self.I(ta.SMA, self.data.Close, self.n_sma)

        .. warning::
            Rolling indicators may front-pad warm-up values with NaNs.
            In this case, the **backtest will only begin on the first bar when
            all declared indicators have non-NaN values** (e.g. bar 201 for a
            strategy that uses a 200-bar MA).
            This can affect results.
        """
        # --- 指标名称格式化 ---
        # 功能层：支持 format 语法——name 中的 {} 会被替换为参数值
        def _format_name(name: str) -> str:
            return name.format(*map(_as_str, args),
                               **dict(zip(kwargs.keys(), map(_as_str, kwargs.values()))))

        # 功能层：自动生成指标名称（如 "SMA(10)"）
        if name is None:
            params = ','.join(filter(None, map(_as_str, chain(args, kwargs.values()))))
            func_name = _as_str(func)
            name = (f'{func_name}({params})' if params else f'{func_name}')
        elif isinstance(name, str):
            name = _format_name(name)
        elif try_(lambda: all(isinstance(item, str) for item in name), False):
            name = [_format_name(item) for item in name]
        else:
            raise TypeError(f'Unexpected `name=` type {type(name)}; expected `str` or '
                            '`Sequence[str]`')

        # 功能层：执行指标计算函数
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f'Indicator "{name}" error. See traceback above.') from e

        # 功能层：如果是 DataFrame，转置为标准形状（列=时间，行=指标线）
        if isinstance(value, pd.DataFrame):
            value = value.values.T

        if value is not None:
            value = try_(lambda: np.asarray(value, order='C'), None)
        is_arraylike = bool(value is not None and value.shape)

        # 功能层：如果用户返回了 (时间, 指标) 形状的数组，自动翻转
        if is_arraylike and np.argmax(value.shape) == 0:
            value = value.T

        # 功能层：验证多维指标的名称数量与维度匹配
        if isinstance(name, list) and (np.atleast_2d(value).shape[0] != len(name)):
            raise ValueError(
                f'Length of `name=` ({len(name)}) must agree with the number '
                f'of arrays the indicator returns ({value.shape[0]}).')

        # 功能层：验证指标长度与数据长度一致
        if not is_arraylike or not 1 <= value.ndim <= 2 or value.shape[-1] != len(self._data.Close):
            raise ValueError(
                'Indicators must return (optionally a tuple of) numpy.arrays of same '
                f'length as `data` (data shape: {self._data.Close.shape}; indicator "{name}" '
                f'shape: {getattr(value, "shape", "")}, returned value: {value})')

        # --- overlay 启发式判定 ---
        # 功能层：如果未指定 overlay，自动判断指标是否是"叠加型"（vs 独立型）
        #          方法：如果大部分指标值在 Close 的 60%-140% 范围内 → 叠加型
        if overlay is None and np.issubdtype(value.dtype, np.number):
            x = value / self._data.Close
            with np.errstate(invalid='ignore'):
                overlay = ((x < 1.4) & (x > .6)).mean() > .6

        # 功能层：包装为 _Indicator 实例（_Array 子类），附加绘制元数据
        value = _Indicator(value, name=name, plot=plot, overlay=overlay,
                           color=color, scatter=scatter,
                           index=self.data.index)
        self._indicators.append(value)  # 注册到策略的指标列表
        return value

    # ====== 抽象方法（用户必须实现） ======
    @abstractmethod
    def init(self):
        """
        Initialize the strategy.
        Override this method.
        Declare indicators (with `backtesting.backtesting.Strategy.I`).
        Precompute what needs to be precomputed or can be precomputed
        in a vectorized fashion before the strategy starts.

        If you extend composable strategies from `backtesting.lib`,
        make sure to call:

            super().init()
        """

    @abstractmethod
    def next(self):
        """
        Main strategy runtime method, called as each new
        `backtesting.backtesting.Strategy.data`
        instance (row; full candlestick bar) becomes available.
        This is the main method where strategy decisions
        upon data precomputed in `backtesting.backtesting.Strategy.init`
        take place.

        If you extend composable strategies from `backtesting.lib`,
        make sure to call:

            super().next()
        """

    # --- 满仓常量 ---
    # 设计层：使用一个特殊的 float 子类，其 __repr__ 返回 ".9999" 而非 "0.999..."
    #          这是一个精致的小技巧，用户友好性优化。
    class __FULL_EQUITY(float):  # noqa: N801
        def __repr__(self): return '.9999'  # noqa: E704
    # 功能层：_FULL_EQUITY = 1 - epsilon ≈ 0.999...，作为 buy/sell 默认 size
    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    # ====== buy —— 多头进场 ======
    def buy(self, *,
            size: float = _FULL_EQUITY,
            limit: Optional[float] = None,
            stop: Optional[float] = None,
            sl: Optional[float] = None,
            tp: Optional[float] = None,
            tag: object = None) -> 'Order':
        """
        Place a new long order and return it. For explanation of parameters, see `Order`
        and its properties.
        Unless you're running `Backtest(..., trade_on_close=True)`,
        market orders are filled on next bar's open,
        whereas other order types (limit, stop-limit, stop-market) are filled when
        the respective conditions are met.

        See `Position.close()` and `Trade.close()` for closing existing positions.

        See also `Strategy.sell()`.
        """
        # 功能层：验证 size 是合法值（0~1 的小数或 >=1 的整数）
        assert 0 < size < 1 or round(size) == size >= 1, \
            "size must be a positive fraction of equity, or a positive whole number of units"
        return self._broker.new_order(size, limit, stop, sl, tp, tag)

    # ====== sell —— 空头进场 ======
    def sell(self, *,
             size: float = _FULL_EQUITY,
             limit: Optional[float] = None,
             stop: Optional[float] = None,
             sl: Optional[float] = None,
             tp: Optional[float] = None,
             tag: object = None) -> 'Order':
        """
        Place a new short order and return it. For explanation of parameters, see `Order`
        and its properties.

        .. caution::
            Keep in mind that `self.sell(size=.1)` doesn't close existing `self.buy(size=.1)`
            trade unless:

            * the backtest was run with `exclusive_orders=True`,
            * the underlying asset price is equal in both cases and
              the backtest was run with `spread = commission = 0`.

            Use `Trade.close()` or `Position.close()` to explicitly exit trades.

        See also `Strategy.buy()`.

        .. note::
            If you merely want to close an existing long position,
            use `Position.close()` or `Trade.close()`.
        """
        assert 0 < size < 1 or round(size) == size >= 1, \
            "size must be a positive fraction of equity, or a positive whole number of units"
        return self._broker.new_order(-size, limit, stop, sl, tp, tag)  # 负数表示空头

    # ====== 属性访问器 ======
    # 设计层：使用 @property 提供简洁的 API 语法。
    #          用户通过 self.position / self.trades 等访问内部状态
    #          而不是直接操作 self._broker。

    @property
    def equity(self) -> float:
        """Current account equity (cash plus assets)."""
        return self._broker.equity

    @property
    def data(self) -> _Data:
        """
        Price data, roughly as passed into
        `backtesting.backtesting.Backtest.__init__`,
        but with two significant exceptions:

        * `data` is _not_ a DataFrame, but a custom structure
          that serves customized numpy arrays for reasons of performance
          and convenience. Besides OHLCV columns, `.index` and length,
          it offers `.pip` property, the smallest price unit of change.
        * Within `backtesting.backtesting.Strategy.init`, `data` arrays
          are available in full length, as passed into
          `backtesting.backtesting.Backtest.__init__`
          (for precomputing indicators and such). However, within
          `backtesting.backtesting.Strategy.next`, `data` arrays are
          only as long as the current iteration, simulating gradual
          price point revelation. In each call of
          `backtesting.backtesting.Strategy.next` (iteratively called by
          `backtesting.backtesting.Backtest` internally),
          the last array value (e.g. `data.Close[-1]`)
          is always the _most recent_ value.
        * If you need data arrays (e.g. `data.Close`) to be indexed
          **Pandas series**, you can call their `.s` accessor
          (e.g. `data.Close.s`). If you need the whole of data
          as a **DataFrame**, use `.df` accessor (i.e. `data.df`).
        """
        return self._data

    @property
    def position(self) -> 'Position':
        """Instance of `backtesting.backtesting.Position`."""
        return self._broker.position

    @property
    def orders(self) -> 'Tuple[Order, ...]':
        """List of orders (see `Order`) waiting for execution."""
        return tuple(self._broker.orders)

    @property
    def trades(self) -> 'Tuple[Trade, ...]':
        """List of active trades (see `Trade`)."""
        return tuple(self._broker.trades)

    @property
    def closed_trades(self) -> 'Tuple[Trade, ...]':
        """List of settled trades (see `Trade`)."""
        return tuple(self._broker.closed_trades)


# ============================================================
# Position —— 当前持仓
# ============================================================
# 上下文层：代表策略当前持有的资产仓位。
#           通过 self.position 访问，支持 bool 判断和 size/pl 查询。
class Position:
    """
    Currently held asset position, available as
    `backtesting.backtesting.Strategy.position` within
    `backtesting.backtesting.Strategy.next`.
    Can be used in boolean contexts, e.g.

        if self.position:
            ...  # we have a position, either long or short
    """
    def __init__(self, broker: '_Broker'):
        self.__broker = broker

    def __bool__(self):
        # 功能层：有仓位 = size != 0
        return self.size != 0

    @property
    def size(self) -> float:
        """Position size in units of asset. Negative if position is short."""
        return sum(trade.size for trade in self.__broker.trades)

    @property
    def pl(self) -> float:
        """Profit (positive) or loss (negative) of the current position in cash units."""
        return sum(trade.pl for trade in self.__broker.trades)

    @property
    def pl_pct(self) -> float:
        """Profit (positive) or loss (negative) of the current position in percent."""
        total_invested = sum(trade.entry_price * abs(trade.size) for trade in self.__broker.trades)
        return (self.pl / total_invested) * 100 if total_invested else 0

    @property
    def is_long(self) -> bool:
        """True if the position is long (position size is positive)."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """True if the position is short (position size is negative)."""
        return self.size < 0

    def close(self, portion: float = 1.):
        """
        Close portion of position by closing `portion` of each active trade. See `Trade.close`.
        """
        for trade in self.__broker.trades:
            trade.close(portion)

    def __repr__(self):
        return f'<Position: {self.size} ({len(self.__broker.trades)} trades)>'


# --- 自定义异常：爆仓 ---
class _OutOfMoneyError(Exception):
    pass


# ============================================================
# Order —— 订单
# ============================================================
# 上下文层：策略通过 buy/sell 创建订单。订单在 Broker 的订单队列中等待撮合。
#           支持市价单、限价单、止损单、止损限价单。
#           每个订单可附带 SL/TP 保护（OCO 一单二挂）。
class Order:
    """
    Place new orders through `Strategy.buy()` and `Strategy.sell()`.
    Query existing orders through `Strategy.orders`.

    When an order is executed or [filled], it results in a `Trade`.

    If you wish to modify aspects of a placed but not yet filled order,
    cancel it and place a new one instead.

    All placed orders are [Good 'Til Canceled].

    [filled]: https://www.investopedia.com/terms/f/fill.asp
    [Good 'Til Canceled]: https://www.investopedia.com/terms/g/gtc.asp
    """
    def __init__(self, broker: '_Broker',
                 size: float,
                 limit_price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 sl_price: Optional[float] = None,
                 tp_price: Optional[float] = None,
                 parent_trade: Optional['Trade'] = None,
                 tag: object = None):
        self.__broker = broker
        assert size != 0
        self.__size = size               # 正=多头 负=空头
        self.__limit_price = limit_price  # 限价（None=市价单）
        self.__stop_price = stop_price    # 止损触发价
        self.__sl_price = sl_price        # 成交后的止盈价
        self.__tp_price = tp_price        # 成交后的止盈价
        self.__parent_trade = parent_trade  # 关联交易（SL/TP 单）
        self.__tag = tag                  # 用户自定义标签

    # 功能层：内部方法——修改私有属性（name mangling: __size → _Order__size）
    # 设计层：Python 的 name mangling 机制（双下划线前缀）
    #          使得私有属性在子类中不会被意外覆盖。
    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def __repr__(self):
        return '<Order {}>'.format(', '.join(f'{param}={try_(lambda: round(value, 5), value)!r}'
                                             for param, value in (
                                                 ('size', self.__size),
                                                 ('limit', self.__limit_price),
                                                 ('stop', self.__stop_price),
                                                 ('sl', self.__sl_price),
                                                 ('tp', self.__tp_price),
                                                 ('contingent', self.is_contingent),
                                                 ('tag', self.__tag),
                                             ) if value is not None))  # noqa: E126

    def cancel(self):
        """Cancel the order."""
        self.__broker.orders.remove(self)      # 从订单队列移除
        trade = self.__parent_trade
        if trade:
            # 功能层：如果该订单是某交易的 SL/TP 单，同步清理
            if self is trade._sl_order:
                trade._replace(sl_order=None)
            elif self is trade._tp_order:
                trade._replace(tp_order=None)
            else:
                pass  # Order placed by Trade.close()

    # --- Order 属性 ---

    @property
    def size(self) -> float:
        """
        Order size (negative for short orders).

        If size is a value between 0 and 1, it is interpreted as a fraction of current
        available liquidity (cash plus `Position.pl` minus used margin).
        A value greater than or equal to 1 indicates an absolute number of units.
        """
        return self.__size

    @property
    def limit(self) -> Optional[float]:
        """
        Order limit price for [limit orders], or None for [market orders],
        which are filled at next available price.

        [limit orders]: https://www.investopedia.com/terms/l/limitorder.asp
        [market orders]: https://www.investopedia.com/terms/m/marketorder.asp
        """
        return self.__limit_price

    @property
    def stop(self) -> Optional[float]:
        """
        Order stop price for [stop-limit/stop-market][_] order,
        otherwise None if no stop was set, or the stop price has already been hit.

        [_]: https://www.investopedia.com/terms/s/stoporder.asp
        """
        return self.__stop_price

    @property
    def sl(self) -> Optional[float]:
        """
        A stop-loss price at which, if set, a new contingent stop-market order
        will be placed upon the `Trade` following this order's execution.
        See also `Trade.sl`.
        """
        return self.__sl_price

    @property
    def tp(self) -> Optional[float]:
        """
        A take-profit price at which, if set, a new contingent limit order
        will be placed upon the `Trade` following this order's execution.
        See also `Trade.tp`.
        """
        return self.__tp_price

    @property
    def parent_trade(self):
        return self.__parent_trade

    @property
    def tag(self):
        """
        Arbitrary value (such as a string) which, if set, enables tracking
        of this order and the associated `Trade` (see `Trade.tag`).
        """
        return self.__tag

    __pdoc__['Order.parent_trade'] = False   # 隐藏内部 API

    # --- 额外属性 ---
    @property
    def is_long(self):
        """True if the order is long (order size is positive)."""
        return self.__size > 0

    @property
    def is_short(self):
        """True if the order is short (order size is negative)."""
        return self.__size < 0

    @property
    def is_contingent(self):
        """
        True for [contingent] orders, i.e. [OCO] stop-loss and take-profit bracket orders
        placed upon an active trade. Remaining contingent orders are canceled when
        their parent `Trade` is closed.

        You can modify contingent orders through `Trade.sl` and `Trade.tp`.

        [contingent]: https://www.investopedia.com/terms/c/contingentorder.asp
        [OCO]: https://www.investopedia.com/terms/o/oco.asp
        """
        return bool((parent := self.__parent_trade) and
                    (self is parent._sl_order or
                     self is parent._tp_order))


# ============================================================
# Trade —— 交易
# ============================================================
# 上下文层：订单成交后产生 Trade。Trade 跟踪从进场到出场的全部信息。
#           活跃交易在 self.trades，已平仓交易在 self.closed_trades。
class Trade:
    """
    When an `Order` is filled, it results in an active `Trade`.
    Find active trades in `Strategy.trades` and closed, settled trades in `Strategy.closed_trades`.
    """
    def __init__(self, broker: '_Broker', size: int, entry_price: float, entry_bar, tag):
        self.__broker = broker
        self.__size = size                 # 正=多头 负=空头
        self.__entry_price = entry_price    # 进场价格
        self.__exit_price: Optional[float] = None  # 出场价格（活跃中为 None）
        self.__entry_bar: int = entry_bar   # 进场 K 线编号
        self.__exit_bar: Optional[int] = None     # 出场 K 线编号
        self.__sl_order: Optional[Order] = None   # 关联的止损单
        self.__tp_order: Optional[Order] = None   # 关联的止盈单
        self.__tag = tag                    # 从订单继承的标签
        self._commissions = 0               # 累计佣金

    def __repr__(self):
        return f'<Trade size={self.__size} time={self.__entry_bar}-{self.__exit_bar or ""} ' \
               f'price={self.__entry_price}-{self.__exit_price or ""} pl={self.pl:.0f}' \
               f'{" tag=" + str(self.__tag) if self.__tag is not None else ""}>'

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def _copy(self, **kwargs):
        # 功能层：创建 Trade 的浅拷贝（用于部分平仓：减少原仓位+创建平仓副本）
        return copy(self)._replace(**kwargs)

    def close(self, portion: float = 1.):
        """Place new `Order` to close `portion` of the trade at next market price."""
        assert 0 < portion <= 1, "portion must be a fraction between 0 and 1"
        # 功能层：计算平仓数量，copysign 保证方向与原交易相反
        size = copysign(max(1, int(round(abs(self.__size) * portion))), -self.__size)
        order = Order(self.__broker, size, parent_trade=self, tag=self.__tag)
        self.__broker.orders.insert(0, order)  # 平仓订单优先处理

    # --- Trade 属性 ---

    @property
    def size(self):
        """Trade size (volume; negative for short trades)."""
        return self.__size

    @property
    def entry_price(self) -> float:
        """Trade entry price."""
        return self.__entry_price

    @property
    def exit_price(self) -> Optional[float]:
        """Trade exit price (or None if the trade is still active)."""
        return self.__exit_price

    @property
    def entry_bar(self) -> int:
        """Candlestick bar index of when the trade was entered."""
        return self.__entry_bar

    @property
    def exit_bar(self) -> Optional[int]:
        """Candlestick bar index of when the trade was exited (or None if active)."""
        return self.__exit_bar

    @property
    def tag(self):
        """
        A tag value inherited from the `Order` that opened this trade.
        This can be used to track trades and apply conditional logic / subgroup analysis.
        """
        return self.__tag

    @property
    def _sl_order(self):
        return self.__sl_order

    @property
    def _tp_order(self):
        return self.__tp_order

    # --- 时间和方向属性 ---
    @property
    def entry_time(self) -> Union[pd.Timestamp, int]:
        """Datetime of when the trade was entered."""
        return self.__broker._data.index[self.__entry_bar]

    @property
    def exit_time(self) -> Optional[Union[pd.Timestamp, int]]:
        """Datetime of when the trade was exited."""
        if self.__exit_bar is None:
            return None
        return self.__broker._data.index[self.__exit_bar]

    @property
    def is_long(self):
        """True if the trade is long (trade size is positive)."""
        return self.__size > 0

    @property
    def is_short(self):
        """True if the trade is short (trade size is negative)."""
        return not self.is_long

    # --- 盈亏属性 ---
    @property
    def pl(self):
        """
        Trade profit (positive) or loss (negative) in cash units.
        Commissions are reflected only after the Trade is closed.
        """
        price = self.__exit_price or self.__broker.last_price
        return (self.__size * (price - self.__entry_price)) - self._commissions

    @property
    def pl_pct(self):
        """Trade profit (positive) or loss (negative) in percent relative to trade entry price."""
        price = self.__exit_price or self.__broker.last_price
        gross_pl_pct = copysign(1, self.__size) * (price / self.__entry_price - 1)
        commission_pct = self._commissions / (abs(self.__size) * self.__entry_price)
        return gross_pl_pct - commission_pct

    @property
    def value(self):
        """Trade total value in cash (volume × price)."""
        price = self.__exit_price or self.__broker.last_price
        return abs(self.__size) * price

    # ====== SL/TP 管理（可读写属性） ======
    # 设计层：使用 @property + @setter 实现属性式 API。
    #          用户写 trade.sl = 100 即可设置/修改止损价。
    #          Python 高级特性：property setter + getter 双方法。

    @property
    def sl(self):
        """
        Stop-loss price at which to close the trade.
        This variable is writable. By assigning it a new price value,
        you create or modify the existing SL order.
        By assigning it `None`, you cancel it.
        """
        return self.__sl_order and self.__sl_order.stop

    @sl.setter
    def sl(self, price: float):
        self.__set_contingent('sl', price)

    @property
    def tp(self):
        """
        Take-profit price at which to close the trade.
        This property is writable. By assigning it a new price value,
        you create or modify the existing TP order.
        By assigning it `None`, you cancel it.
        """
        return self.__tp_order and self.__tp_order.limit

    @tp.setter
    def tp(self, price: float):
        self.__set_contingent('tp', price)

    # 功能层：SL/TP 设置的统一实现
    def __set_contingent(self, type, price):
        assert type in ('sl', 'tp')
        assert price is None or 0 < price < np.inf, f'Make sure 0 < price < inf! price: {price}'
        attr = f'_{self.__class__.__qualname__}__{type}_order'
        order: Order = getattr(self, attr)
        if order:
            order.cancel()  # 功能层：取消旧的 SL/TP 订单
        if price:
            # 功能层：创建新的 SL（止损市价单）或 TP（止盈限价单）
            kwargs = {'stop': price} if type == 'sl' else {'limit': price}
            order = self.__broker.new_order(-self.size, trade=self, tag=self.tag, **kwargs)
            setattr(self, attr, order)


# ============================================================
# _Broker —— 内部经纪商/撮合引擎
# ============================================================
# 上下文层：_Broker 是回测引擎的"心脏"。
#           负责订单撮合、交易状态管理、保证金计算、权益追踪。
#           这是一个内部类——用户不直接接触，但它的逻辑决定了回测行为。
# 设计层：状态机模式——每个 K 线周期内处理订单队列，
#          更新持仓、权益、并处理保证金不足等边界情况。
class _Broker:
    def __init__(self, *, data, cash, spread, commission, margin,
                 trade_on_close, hedging, exclusive_orders, index):
        assert cash > 0, f"cash should be > 0, is {cash}"
        assert 0 < margin <= 1, f"margin should be between 0 and 1, is {margin}"
        self._data: _Data = data
        self._cash = cash  # 当前可用现金

        # --- 佣金配置 ---
        # 设计层：支持三种佣金模型——固定+比例、纯比例、自定义函数
        if callable(commission):
            self._commission = commission
        else:
            try:
                self._commission_fixed, self._commission_relative = commission  # 解包 (固定, 比例)
            except TypeError:
                self._commission_fixed, self._commission_relative = 0, commission  # 纯比例
            assert self._commission_fixed >= 0, 'Need fixed cash commission in $ >= 0'
            assert -.1 <= self._commission_relative < .1, \
                ("commission should be between -10% "
                 f"(e.g. market-maker's rebates) and 10% (fees), is {self._commission_relative}")
            self._commission = self._commission_func

        self._spread = spread                      # 买卖价差率
        self._leverage = 1 / margin                # 杠杆倍数（保证金倒数）
        self._trade_on_close = trade_on_close       # 市价单是否按收盘价成交
        self._hedging = hedging                     # 是否允许对冲（同时持有长短仓）
        self._exclusive_orders = exclusive_orders   # 新订单是否自动平旧仓

        self._equity = np.tile(np.nan, len(index))  # 权益曲线（每根 K 线一个值）
        self.orders: List[Order] = []                # 挂单队列
        self.trades: List[Trade] = []                # 活跃交易列表
        self.position = Position(self)               # 持仓状态
        self.closed_trades: List[Trade] = []         # 已平仓交易历史

    def _commission_func(self, order_size, price):
        # 功能层：固定+比例佣金计算公式
        return self._commission_fixed + abs(order_size) * price * self._commission_relative

    def __repr__(self):
        return f'<Broker: {self._cash:.0f}{self.position.pl:+.1f} ({len(self.trades)} trades)>'

    # ====== new_order —— 创建新订单 ======
    def new_order(self,
                  size: float,
                  limit: Optional[float] = None,
                  stop: Optional[float] = None,
                  sl: Optional[float] = None,
                  tp: Optional[float] = None,
                  tag: object = None,
                  *,
                  trade: Optional[Trade] = None) -> Order:
        """
        Argument size indicates whether the order is long or short
        """
        size = float(size)
        stop = stop and float(stop)
        limit = limit and float(limit)
        sl = sl and float(sl)
        tp = tp and float(tp)

        is_long = size > 0
        assert size != 0, size
        adjusted_price = self._adjusted_price(size)  # 含价差的调整后价格

        # 功能层：验证价格逻辑（多头: SL < 入场价 < TP；空头: TP < 入场价 < SL）
        if is_long:
            if not (sl or -np.inf) < (limit or stop or adjusted_price) < (tp or np.inf):
                raise ValueError(
                    "Long orders require: "
                    f"SL ({sl}) < LIMIT ({limit or stop or adjusted_price}) < TP ({tp})")
        else:
            if not (tp or -np.inf) < (limit or stop or adjusted_price) < (sl or np.inf):
                raise ValueError(
                    "Short orders require: "
                    f"TP ({tp}) < LIMIT ({limit or stop or adjusted_price}) < SL ({sl})")

        order = Order(self, size, limit, stop, sl, tp, trade, tag)

        # --- exclusive_orders 模式：新订单自动平旧仓 ---
        if not trade:
            if self._exclusive_orders:
                for o in self.orders:
                    if not o.is_contingent:
                        o.cancel()          # 取消所有非 contingent 挂单
                for t in self.trades:
                    t.close()               # 平掉所有活跃交易

        # 功能层：将订单插入队列（SL 订单优先出现——insert(0, ...)）
        self.orders.insert(0 if trade and stop else len(self.orders), order)

        return order

    @property
    def last_price(self) -> float:
        """ Price at the last (current) close. """
        return self._data.Close[-1]

    def _adjusted_price(self, size=None, price=None) -> float:
        """
        Long/short `price`, adjusted for spread.
        In long positions, the adjusted price is a fraction higher, and vice versa.
        """
        return (price or self.last_price) * (1 + copysign(self._spread, size))

    @property
    def equity(self) -> float:
        # 功能层：权益 = 现金 + 所有活跃交易的浮动盈亏
        return self._cash + sum(trade.pl for trade in self.trades)

    @property
    def margin_available(self) -> float:
        # 功能层：可用保证金 = max(0, 权益 - 已用保证金)
        margin_used = sum(trade.value / self._leverage for trade in self.trades)
        return max(0, self.equity - margin_used)

    # ====== next —— 每根 K 线的撮合循环 ======
    def next(self):
        i = self._i = len(self._data) - 1
        self._process_orders()  # 核心：处理所有挂单

        equity = self.equity
        self._equity[i] = equity  # 记录权益曲线

        # 功能层：爆仓检测——权益<=0 则清空所有仓位并停止回测
        if equity <= 0:
            assert self.margin_available <= 0
            for trade in self.trades:
                self._close_trade(trade, self._data.Close[-1], i)
            self._cash = 0
            self._equity[i:] = 0
            raise _OutOfMoneyError  # 抛出异常，Backtest.run 会捕获

    # ====== _process_orders —— 订单撮合（最复杂的核心逻辑） ======
    # 上下文层：这是整个回测引擎中逻辑最密集的方法。
    #           每根 K 线调用一次，遍历订单队列并判断是否满足成交条件。
    def _process_orders(self):
        data = self._data
        open, high, low = data.Open[-1], data.High[-1], data.Low[-1]  # 当前 K 线的开/高/低
        reprocess_orders = False  # 是否需要重新处理（新 SL/TP 单可能在同 K 线触发）

        for order in list(self.orders):  # type: Order

            # 如果订单已被移除（例如被其他逻辑取消），跳过
            if order not in self.orders:
                continue

            # --- Step 1: 检查止损是否触发 ---
            stop_price = order.stop
            if stop_price:
                # 功能层：多头止损=价格跌破stop；空头止损=价格涨破stop
                is_stop_hit = ((high >= stop_price) if order.is_long else (low <= stop_price))
                if not is_stop_hit:
                    continue
                order._replace(stop_price=None)  # 止损触发后，转为市价/限价单

            # --- Step 2: 检查限价是否可达 ---
            if order.limit:
                is_limit_hit = low <= order.limit if order.is_long else high >= order.limit
                # 功能层：悲观假设——同一 K 线内，限价在止损之前触及
                is_limit_hit_before_stop = (is_limit_hit and
                                            (order.limit <= (stop_price or -np.inf)
                                             if order.is_long
                                             else order.limit >= (stop_price or np.inf)))
                if not is_limit_hit or is_limit_hit_before_stop:
                    continue

                # 功能层：成交价 = min(stop或open, limit) / max(stop或open, limit)
                price = (min(stop_price or open, order.limit)
                         if order.is_long else
                         max(stop_price or open, order.limit))
            else:
                # 功能层：市价单——按开盘价（或 trade_on_close 模式按前收盘价）成交
                prev_close = data.Close[-2]
                price = prev_close if self._trade_on_close and not order.is_contingent else open
                if stop_price:
                    # 止损市价单：确保价格至少超出止损价
                    price = max(price, stop_price) if order.is_long else min(price, stop_price)

            # --- Step 3: 确定成交的时间索引 ---
            is_market_order = not order.limit and not stop_price
            time_index = (
                (self._i - 1)
                if is_market_order and self._trade_on_close and not order.is_contingent else
                self._i)

            # --- Step 4: contingent 订单（SL/TP）处理 ---
            if order.parent_trade:
                trade = order.parent_trade
                _prev_size = trade.size
                size = copysign(min(abs(_prev_size), abs(order.size)), order.size)
                if trade in self.trades:
                    self._reduce_trade(trade, price, size, time_index)
                    assert order.size != -_prev_size or trade not in self.trades
                    if price == stop_price:
                        trade._sl_order._replace(stop_price=stop_price)
                if order in (trade._sl_order, trade._tp_order):
                    assert order.size == -trade.size
                    assert order not in self.orders
                else:
                    assert abs(_prev_size) >= abs(size) >= 1
                    self.orders.remove(order)
                continue

            # --- Step 5: 独立订单（新建仓位） ---
            adjusted_price = self._adjusted_price(order.size, price)
            adjusted_price_plus_commission = \
                adjusted_price + self._commission(order.size, price) / abs(order.size)

            # 功能层：比例仓位——将 0~1 的 size 转换为实际股数
            size = order.size
            if -1 < size < 1:
                size = copysign(int((self.margin_available * self._leverage * abs(size))
                                    // adjusted_price_plus_commission), size)
                if not size:
                    # 功能层：资金/保证金不足以购买单股——经纪商取消订单
                    warnings.warn(
                        f'time={self._i}: Broker canceled the relative-sized order due to insufficient margin '
                        f'(equity={self.equity:.2f}, margin_available={self.margin_available:.2f}).',
                        category=UserWarning)
                    self.orders.remove(order)
                    continue
            assert size == round(size)
            need_size = int(size)

            # --- 对冲模式关闭时：FIFO 平掉反向仓位 ---
            if not self._hedging:
                for trade in list(self.trades):
                    if trade.is_long == order.is_long:
                        continue  # 同向仓位，跳过
                    assert trade.size * order.size < 0

                    if abs(need_size) >= abs(trade.size):
                        self._close_trade(trade, price, time_index)  # 完全平掉反向仓位
                        need_size += trade.size
                    else:
                        self._reduce_trade(trade, price, need_size, time_index)  # 部分平仓
                        need_size = 0

                    if not need_size:
                        break

            # --- 保证金不足检查 ---
            if abs(need_size) * adjusted_price_plus_commission > \
                    self.margin_available * self._leverage:
                warnings.warn(
                    f'time={self._i}: Broker canceled the order due to insufficient margin '
                    f'(equity={self.equity:.2f}, margin_available={self.margin_available:.2f}).',
                    category=UserWarning)
                self.orders.remove(order)
                continue

            # --- 开新仓位 ---
            if need_size:
                self._open_trade(adjusted_price, need_size, order.sl, order.tp,
                                time_index, order.tag)

                # 功能层：SL/TP 单被加入队列，需要重新处理以检查同 K 线触发
                if order.sl or order.tp:
                    if is_market_order:
                        reprocess_orders = True
                    elif stop_price and not order.limit and order.tp and (
                            (order.is_long and order.tp <= high and (order.sl or -np.inf) < low) or
                            (order.is_short and order.tp >= low and (order.sl or np.inf) > high)):
                        reprocess_orders = True
                    elif (low <= (order.sl or -np.inf) <= high or
                          low <= (order.tp or -np.inf) <= high):
                        warnings.warn(
                            f"({data.index[-1]}) A contingent SL/TP order would execute in the "
                            "same bar its parent stop/limit order was turned into a trade. "
                            "Since we can't assert the precise intra-candle "
                            "price movement, the affected SL/TP order will instead be executed on "
                            "the next (matching) price/bar, making the result (of this trade) "
                            "somewhat dubious. "
                            "See https://github.com/kernc/backtesting.py/issues/119",
                            UserWarning)

            # 订单处理完毕，从队列中移除
            self.orders.remove(order)

        # 如果新产生的 SL/TP 单可能在同一根 K 线触发，递归处理
        if reprocess_orders:
            self._process_orders()

    # ====== _reduce_trade —— 部分平仓 ======
    def _reduce_trade(self, trade: Trade, price: float, size: float, time_index: int):
        assert trade.size * size < 0
        assert abs(trade.size) >= abs(size)

        size_left = trade.size + size
        assert size_left * trade.size >= 0
        if not size_left:
            close_trade = trade
        else:
            # 功能层：减少原仓位...
            trade._replace(size=size_left)
            if trade._sl_order:
                trade._sl_order._replace(size=-trade.size)
            if trade._tp_order:
                trade._tp_order._replace(size=-trade.size)

            # ...然后创建一个减少部分的副本并入平仓流程
            close_trade = trade._copy(size=-size, sl_order=None, tp_order=None)
            self.trades.append(close_trade)

        self._close_trade(close_trade, price, time_index)

    # ====== _close_trade —— 完全平仓 ======
    def _close_trade(self, trade: Trade, price: float, time_index: int):
        self.trades.remove(trade)
        if trade._sl_order:
            self.orders.remove(trade._sl_order)
        if trade._tp_order:
            self.orders.remove(trade._tp_order)

        closed_trade = trade._replace(exit_price=price, exit_bar=time_index)
        self.closed_trades.append(closed_trade)
        # 功能层：出场时再扣一次佣金（共两次）
        commission = self._commission(trade.size, price)
        self._cash += trade.pl - commission
        # 功能层：保存累计佣金到 Trade 实例（供统计使用）
        trade_open_commission = self._commission(closed_trade.size, closed_trade.entry_price)
        closed_trade._commissions = commission + trade_open_commission

    # ====== _open_trade —— 新建仓位 ======
    def _open_trade(self, price: float, size: int,
                    sl: Optional[float], tp: Optional[float], time_index: int, tag):
        trade = Trade(self, size, price, time_index, tag)
        self.trades.append(trade)
        self._cash -= self._commission(size, price)  # 功能层：进场时扣佣金
        if tp:
            trade.tp = tp   # 创建止盈单
        if sl:
            trade.sl = sl   # 创建止损单


# ============================================================
# Backtest —— 回测引擎（用户入口）
# ============================================================
# 上下文层：这是用户直接使用的类。创建 Backtest 实例，
#           调用 run() 执行回测，调用 optimize() 参数优化，调用 plot() 可视化。
class Backtest:
    """
    Backtest a particular (parameterized) strategy on particular data.

    Initialize a backtest. Requires data and a strategy to test.
    After initialization, you can call method
    `backtesting.backtesting.Backtest.run` to run a backtest
    instance, or `backtesting.backtesting.Backtest.optimize` to
    optimize it.

    `data` is a `pd.DataFrame` with columns:
    `Open`, `High`, `Low`, `Close`, and (optionally) `Volume`.
    If any columns are missing, set them to what you have available, e.g.

        df['Open'] = df['High'] = df['Low'] = df['Close']

    The passed data frame can contain additional columns that
    can be used by the strategy (e.g. sentiment info).
    DataFrame index can be either a datetime index (timestamps)
    or a monotonic range index (i.e. a sequence of periods).

    `strategy` is a `backtesting.backtesting.Strategy`
    _subclass_ (not an instance).

    `cash` is the initial cash to start with.

    `spread` is the constant bid-ask spread rate (relative to the price).

    `commission` is the commission rate. E.g. if your broker's commission
    is 1% of order value, set commission to `0.01`.
    The commission is applied twice: at trade entry and at trade exit.

    `margin` is the required margin (ratio) of a leveraged account.

    If `trade_on_close` is `True`, market orders will be filled
    with respect to the current bar's closing price instead of the
    next bar's open.

    If `hedging` is `True`, allow trades in both directions simultaneously.
    If `False`, the opposite-facing orders first close existing trades in
    a [FIFO] manner.

    If `exclusive_orders` is `True`, each new order auto-closes the previous
    trade/position, making at most a single trade (long or short) in effect
    at each time.

    If `finalize_trades` is `True`, the trades that are still
    [active and ongoing] at the end of the backtest will be closed on
    the last bar and will contribute to the computed backtest statistics.

    [FIFO]: https://www.investopedia.com/terms/n/nfa-compliance-rule-2-43b.asp
    [active and ongoing]: https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy.trades
    """  # noqa: E501
    def __init__(self,
                 data: pd.DataFrame,
                 strategy: Type[Strategy],
                 *,
                 cash: float = 10_000,
                 spread: float = .0,
                 commission: Union[float, Tuple[float, float]] = .0,
                 margin: float = 1.,
                 trade_on_close=False,
                 hedging=False,
                 exclusive_orders=False,
                 finalize_trades=False,
                 ):
        # --- 输入验证 ---
        if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
            raise TypeError('`strategy` must be a Strategy sub-type')
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas.DataFrame with columns")
        if not isinstance(spread, Number):
            raise TypeError('`spread` must be a float value, percent of entry order price')
        if not isinstance(commission, (Number, tuple)) and not callable(commission):
            raise TypeError('`commission` must be a float percent of order value, '
                            'a tuple of `(fixed, relative)` commission, '
                            'or a function that takes `(order_size, price)`'
                            'and returns commission dollar value')

        data = data.copy(deep=False)  # 浅拷贝，避免修改用户传入的原始数据

        # --- 智能日期索引转换 ---
        if (not isinstance(data.index, pd.DatetimeIndex) and
            not isinstance(data.index, pd.RangeIndex) and
            (data.index.is_numeric() and
             (data.index > pd.Timestamp('1975').timestamp()).mean() > .8)):
            try:
                data.index = pd.to_datetime(data.index, infer_datetime_format=True)
            except ValueError:
                pass

        # --- 数据完整性验证 ---
        if 'Volume' not in data:
            data['Volume'] = np.nan   # 成交量可选

        if len(data) == 0:
            raise ValueError('OHLC `data` is empty')
        if len(data.columns.intersection({'Open', 'High', 'Low', 'Close', 'Volume'})) != 5:
            raise ValueError("`data` must be a pandas.DataFrame with columns "
                             "'Open', 'High', 'Low', 'Close', and (optionally) 'Volume'")
        if data[['Open', 'High', 'Low', 'Close']].isnull().values.any():
            raise ValueError('Some OHLC values are missing (NaN).')

        # 功能层：价格超过初始现金的警告（非分数股交易）
        if np.any(data['Close'] > cash):
            warnings.warn('Some prices are larger than initial cash value. Note that fractional '
                          'trading is not supported by this class.')

        if not data.index.is_monotonic_increasing:
            warnings.warn('Data index is not sorted in ascending order. Sorting.')
            data = data.sort_index()
        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.warn('Data index is not datetime. Assuming simple periods, '
                          'but `pd.DateTimeIndex` is advised.')

        self._data: pd.DataFrame = data
        # 设计层：使用 partial 延迟 _Broker 的创建。
        #         每次 run() 会重新调用 self._broker() 创建新的 Broker 实例。
        self._broker = partial(
            _Broker, cash=cash, spread=spread, commission=commission, margin=margin,
            trade_on_close=trade_on_close, hedging=hedging,
            exclusive_orders=exclusive_orders, index=data.index,
        )
        self._strategy = strategy          # 策略类（非实例）
        self._results: Optional[pd.Series] = None  # 最近一次 run 的结果
        self._finalize_trades = bool(finalize_trades)

    # ====== run —— 执行回测 ======
    # 上下文层：这是用户调用 bt.run() 的入口。
    #           它是框架的"导演"——创建数据、Broker、策略实例，
    #           然后驱动逐 K 线回测主循环。
    def run(self, **kwargs) -> pd.Series:
        """
        Run the backtest. Returns `pd.Series` with results and statistics.

        Keyword arguments are interpreted as strategy parameters.

            >>> Backtest(GOOG, SmaCross).run()
            Start                     2004-08-19 00:00:00
            End                       2013-03-01 00:00:00
            Duration                   3116 days 00:00:00
            Exposure Time [%]                    96.74115
            Equity Final [$]                     51422.99
            Equity Peak [$]                      75787.44
            Return [%]                           414.2299
            Buy & Hold Return [%]               703.45824
            Return (Ann.) [%]                    21.18026
            Volatility (Ann.) [%]                36.49391
            CAGR [%]                             14.15984
            Sharpe Ratio                          0.58038
            Sortino Ratio                         1.08479
            Calmar Ratio                          0.44144
            Alpha [%]                           394.37391
            Beta                                  0.03803
            Max. Drawdown [%]                   -47.98013
            Avg. Drawdown [%]                    -5.92585
            Max. Drawdown Duration      584 days 00:00:00
            Avg. Drawdown Duration       41 days 00:00:00
            # Trades                                   66
            Win Rate [%]                          46.9697
            Best Trade [%]                       53.59595
            Worst Trade [%]                     -18.39887
            Avg. Trade [%]                        2.53172
            Max. Trade Duration         183 days 00:00:00
            Avg. Trade Duration          46 days 00:00:00
            Profit Factor                         2.16795
            Expectancy [%]                        3.27481
            SQN                                   1.07662
            Kelly Criterion                       0.15187
            _strategy                            SmaCross
            _equity_curve                           Eq...
            _trades                       Size  EntryB...
            dtype: object

        .. warning::
            You may obtain different results for different strategy parameters.
            E.g. if you use 50- and 200-bar SMA, the trading simulation will
            begin on bar 201. The actual length of delay is equal to the lookback
            period of the `Strategy.I` indicator which lags the most.
            Obviously, this can affect results.
        """
        data = _Data(self._data.copy(deep=False))     # 创建数据访问器
        broker: _Broker = self._broker(data=data)     # 创建 Broker（partial 展开）
        strategy: Strategy = self._strategy(broker, data, kwargs)  # 创建策略实例

        strategy.init()           # 阶段 1：策略初始化（预计算指标）
        data._update()            # 阶段 1b：策略可能在 init 中添加了新列

        indicator_attrs = _strategy_indicators(strategy)  # 收集指标属性

        # 功能层：跳过指标预热期（+1 保证至少有两条数据可用）
        start = 1 + _indicator_warmup_nbars(strategy)

        # 抑制 numpy 对 NaN 比较的无效值警告
        with np.errstate(invalid='ignore'):

            # ====== 主循环：逐根 K 线推进 ======
            # 设计层：这是事件驱动回测的核心循环。
            #          每根 K 线依次执行：揭示数据 → 处理订单 → 调用策略
            for i in _tqdm(range(start, len(self._data)), desc=self.run.__qualname__,
                           unit='bar', mininterval=2, miniters=100):
                # Step 1: 将数据"逐步揭示"（仅暴露到 i 为止的数据）
                data._set_length(i + 1)
                for attr, indicator in indicator_attrs:
                    setattr(strategy, attr, indicator[..., :i + 1])

                # Step 2: Broker 处理订单撮合（可能抛出 _OutOfMoneyError）
                try:
                    broker.next()
                except _OutOfMoneyError:
                    break

                # Step 3: 调用策略的交易决策（next tick, a moment before bar close）
                strategy.next()
            else:
                # --- 循环正常结束后的收尾处理 ---
                if self._finalize_trades is True:
                    # 功能层：平掉所有未结束的交易
                    for trade in reversed(broker.trades):
                        trade.close()
                    if start < len(self._data):
                        try_(broker.next, exception=_OutOfMoneyError)
                elif len(broker.trades):
                    # 功能层：警告用户存在未平仓交易
                    warnings.warn(
                        'Some trades remain open at the end of backtest. Use '
                        '`Backtest(..., finalize_trades=True)` to close them and '
                        'include them in stats.', stacklevel=2)

            # 恢复数据到完整长度
            data._set_length(len(self._data))

            # 功能层：构建权益曲线（后向填充 NaN，如预热期间的缺失值）
            equity = pd.Series(broker._equity).bfill().fillna(broker._cash).values
            # 功能层：计算最终统计结果
            self._results = compute_stats(
                trades=broker.closed_trades,
                equity=equity,
                ohlc_data=self._data,
                risk_free_rate=0.0,
                strategy_instance=strategy,
            )

        return self._results

    # ====== optimize —— 参数优化 ======
    # 上下文层：自动搜索最优策略参数组合。
    #           支持网格搜索（exhaustive）和 SAMBO 贝叶斯优化两种方法。
    def optimize(self, *,
                 maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
                 method: str = 'grid',
                 max_tries: Optional[Union[int, float]] = None,
                 constraint: Optional[Callable[[dict], bool]] = None,
                 return_heatmap: bool = False,
                 return_optimization: bool = False,
                 random_state: Optional[int] = None,
                 **kwargs) -> Union[pd.Series,
                                    Tuple[pd.Series, pd.Series],
                                    Tuple[pd.Series, pd.Series, dict]]:
        """
        Optimize strategy parameters to an optimal combination.
        Returns result `pd.Series` of the best run.

        `maximize` is a string key from the
        `backtesting.backtesting.Backtest.run`-returned results series,
        or a function that accepts this series object and returns a number;
        the higher the better. By default, the method maximizes
        Van Tharp's [System Quality Number](https://google.com/search?q=System+Quality+Number).

        `method` is the optimization method. Currently two methods are supported:

        * `"grid"` which does an exhaustive (or randomized) search over the
          cartesian product of parameter combinations, and
        * `"sambo"` which finds close-to-optimal strategy parameters using
          [model-based optimization], making at most `max_tries` evaluations.

        [model-based optimization]: https://sambo-optimization.github.io

        `max_tries` is the maximal number of strategy runs to perform.
        If `method="grid"`, this results in randomized grid search.

        `constraint` is a function that accepts a dict-like object of
        parameters (with values) and returns `True` when the combination
        is admissible to test with.

        If `return_heatmap` is `True`, besides returning the result
        series, an additional `pd.Series` is returned with a multiindex
        of all admissible parameter combinations.

        Additional keyword arguments represent strategy arguments with
        list-like collections of possible values.
        """
        if not kwargs:
            raise ValueError('Need some strategy parameters to optimize')

        # --- maximize 参数处理 ---
        maximize_key = None
        if isinstance(maximize, str):
            maximize_key = str(maximize)
            if maximize not in dummy_stats().index:
                raise ValueError('`maximize`, if str, must match a key in pd.Series '
                                 'result of backtest.run()')

            def maximize(stats: pd.Series, _key=maximize):
                return stats[_key]

        elif not callable(maximize):
            raise TypeError('`maximize` must be str (a field of backtest.run() result '
                            'Series) or a function that accepts result Series '
                            'and returns a number; the higher the better')
        assert callable(maximize), maximize

        # --- constraint 参数处理 ---
        have_constraint = bool(constraint)
        if constraint is None:

            def constraint(_):
                return True

        elif not callable(constraint):
            raise TypeError("`constraint` must be a function that accepts a dict "
                            "of strategy parameters and returns a bool whether "
                            "the combination of parameters is admissible or not")
        assert callable(constraint), constraint

        # --- method 兼容处理 ---
        if method == 'skopt':
            method = 'sambo'
            warnings.warn('`Backtest.optimize(method="skopt")` is deprecated. Use `method="sambo"`.',
                          DeprecationWarning, stacklevel=2)
        if return_optimization and method != 'sambo':
            raise ValueError("return_optimization=True only valid if method='sambo'")

        # --- 辅助：将值转为元组 ---
        def _tuple(x):
            return x if isinstance(x, Sequence) and not isinstance(x, str) else (x,)

        for k, v in kwargs.items():
            if len(_tuple(v)) == 0:
                raise ValueError(f"Optimization variable '{k}' is passed no "
                               f"optimization values: {k}={v}")

        # --- AttrDict：字典但支持属性访问（p.sma1 而非 p['sma1']） ---
        # 设计层：为用户约束函数提供更自然的参数访问方式。
        class AttrDict(dict):
            def __getattr__(self, item):
                return self[item]

        # --- 计算网格大小 ---
        def _grid_size():
            size = int(np.prod([len(_tuple(v)) for v in kwargs.values()]))
            if size < 10_000 and have_constraint:
                size = sum(1 for p in product(*(zip(repeat(k), _tuple(v))
                                                for k, v in kwargs.items()))
                           if constraint(AttrDict(p)))
            return size

        # ====== _optimize_grid —— 网格搜索 ======
        def _optimize_grid() -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
            rand = default_rng(random_state).random
            grid_frac = (1 if max_tries is None else
                         max_tries if 0 < max_tries <= 1 else
                         max_tries / _grid_size())
            param_combos = [dict(params)
                            for params in (AttrDict(params)
                                           for params in product(*(zip(repeat(k), _tuple(v))
                                                                   for k, v in kwargs.items())))
                            if constraint(params)
                            and rand() <= grid_frac]
            if not param_combos:
                raise ValueError('No admissible parameter combinations to test')

            if len(param_combos) > 300:
                warnings.warn(f'Searching for best of {len(param_combos)} configurations.')

            heatmap = pd.Series(np.nan,
                                name=maximize_key,
                                index=pd.MultiIndex.from_tuples(
                                    [p.values() for p in param_combos],
                                    names=next(iter(param_combos)).keys()))

            # 功能层：多进程并行执行每个参数组合的回测
            from . import Pool
            with Pool() as pool, \
                    SharedMemoryManager() as smm:
                with patch(self, '_data', None):
                    bt = copy(self)  # bt._data will be reassigned in _mp_task worker
                results = _tqdm(
                    pool.imap(Backtest._mp_task,
                              ((bt, smm.df2shm(self._data), params_batch)
                               for params_batch in _batch(param_combos))),
                    total=len(param_combos),
                    desc='Backtest.optimize'
                )
                for param_batch, result in zip(_batch(param_combos), results):
                    for params, stats in zip(param_batch, result):
                        if stats is not None:
                            heatmap[tuple(params.values())] = maximize(stats)

            # 功能层：如果没有产生任何交易，使用第一组参数运行以获得空结果
            if pd.isnull(heatmap).all():
                stats = self.run(**param_combos[0])
            else:
                # 功能层：选择最大化目标的最佳参数组合
                best_params = heatmap.idxmax(skipna=True)
                stats = self.run(**dict(zip(heatmap.index.names, best_params)))

            if return_heatmap:
                return stats, heatmap
            return stats

        # ====== _optimize_sambo —— SAMBO 贝叶斯优化 ======
        def _optimize_sambo() -> Union[pd.Series,
                                       Tuple[pd.Series, pd.Series],
                                       Tuple[pd.Series, pd.Series, dict]]:
            try:
                import sambo
            except ImportError:
                raise ImportError("Need package 'sambo' for method='sambo'. pip install sambo") from None

            nonlocal max_tries
            max_tries = (200 if max_tries is None else
                         max(1, int(max_tries * _grid_size())) if 0 < max_tries <= 1 else
                         max_tries)

            # 功能层：构建优化维度
            dimensions = []
            for key, values in kwargs.items():
                values = np.asarray(values)
                if values.dtype.kind in 'mM':
                    values = values.astype(np.int64)
                if values.dtype.kind in 'iumM':
                    dimensions.append((values.min(), values.max() + 1))
                elif values.dtype.kind == 'f':
                    dimensions.append((values.min(), values.max()))
                else:
                    dimensions.append(values.tolist())

            # 功能层：LRU 缓存避免对相同参数重复运行回测
            @lru_cache()
            def memoized_run(tup):
                nonlocal maximize, self
                stats = self.run(**dict(tup))
                return -maximize(stats)  # SAMBO 最小化 → 目标取负

            progress = iter(_tqdm(repeat(None), total=max_tries, leave=False,
                                  desc=self.optimize.__qualname__, mininterval=2))
            _names = tuple(kwargs.keys())

            def objective_function(x):
                nonlocal progress, memoized_run, constraint, _names
                next(progress)
                value = memoized_run(tuple(zip(_names, x)))
                return 0 if np.isnan(value) else value

            def cons(x):
                nonlocal constraint, _names
                return constraint(AttrDict(zip(_names, x)))

            res = sambo.minimize(
                fun=objective_function,
                bounds=dimensions,
                constraints=cons,
                max_iter=max_tries,
                method='sceua',
                rng=random_state)

            stats = self.run(**dict(zip(kwargs.keys(), res.x)))
            output = [stats]

            if return_heatmap:
                heatmap = pd.Series(dict(zip(map(tuple, res.xv), -res.funv)),
                                    name=maximize_key)
                heatmap.index.names = kwargs.keys()
                heatmap.sort_index(inplace=True)
                output.append(heatmap)

            if return_optimization:
                output.append(res)

            return stats if len(output) == 1 else tuple(output)

        # --- 分发到具体优化方法 ---
        if method == 'grid':
            output = _optimize_grid()
        elif method in ('sambo', 'skopt'):
            output = _optimize_sambo()
        else:
            raise ValueError(f"Method should be 'grid' or 'sambo', not {method!r}")
        return output

    # ====== _mp_task —— 多进程回测任务（静态方法） ======
    # 设计层：静态方法 + 共享内存传输大 DataFrame。
    #          避免通过 pickle 传递数据，显著提升多进程优化效率。
    @staticmethod
    def _mp_task(arg):
        bt, data_shm, params_batch = arg
        bt._data, shm = SharedMemoryManager.shm2df(data_shm)
        try:
            return [stats.filter(regex='^[^_]') if stats['# Trades'] else None
                    for stats in (bt.run(**params)
                                  for params in params_batch)]
        finally:
            for shmem in shm:
                shmem.close()

    # ====== plot —— 图表绘制 ======
    def plot(self, *, results: pd.Series = None, filename=None, plot_width=None,
             plot_equity=True, plot_return=False, plot_pl=True,
             plot_volume=True, plot_drawdown=False, plot_trades=True,
             smooth_equity=False, relative_equity=True,
             superimpose: Union[bool, str] = True,
             resample=True, reverse_indicators=False,
             show_legend=True, open_browser=True):
        """
        Plot the progression of the last backtest run.

        If `results` is provided, it should be a particular result
        `pd.Series` such as returned by
        `backtesting.backtesting.Backtest.run` or
        `backtesting.backtesting.Backtest.optimize`, otherwise the last
        run's results are used.

        `filename` is the path to save the interactive HTML plot to.

        `plot_width` is the width of the plot in pixels.

        If `plot_equity` is `True`, the resulting plot will contain
        an equity graph section.

        If `plot_return` is `True`, the resulting plot will contain
        a cumulative return graph section.

        If `plot_pl` is `True`, the resulting plot will contain
        a profit/loss (P/L) indicator section.

        If `plot_volume` is `True`, the resulting plot will contain
        a trade volume section.

        If `plot_drawdown` is `True`, the resulting plot will contain
        a separate drawdown graph section.

        If `plot_trades` is `True`, the stretches between trade entries
        and trade exits are marked.

        If `smooth_equity` is `True`, the equity graph will be
        interpolated between fixed points at trade closing times.

        If `relative_equity` is `True`, scale and label equity graph axis
        with return percent, not absolute cash-equivalent values.

        If `superimpose` is `True`, superimpose larger-timeframe candlesticks
        over the original candlestick chart.

        If `resample` is `True`, the OHLC data is resampled to limit
        the number of candles to 10_000 for performance.

        If `reverse_indicators` is `True`, the indicators below the OHLC chart
        are plotted in reverse order of declaration.

        If `show_legend` is `True`, the resulting plot graphs will contain
        labeled legends.

        If `open_browser` is `True`, the resulting `filename` will be
        opened in the default web browser.
        """
        if results is None:
            if self._results is None:
                raise RuntimeError('First issue `backtest.run()` to obtain results.')
            results = self._results

        # 功能层：委托给 _plotting.plot 生成 Bokeh HTML 图表
        return plot(
            results=results,
            df=self._data,
            indicators=results._strategy._indicators,
            filename=filename,
            plot_width=plot_width,
            plot_equity=plot_equity,
            plot_return=plot_return,
            plot_pl=plot_pl,
            plot_volume=plot_volume,
            plot_drawdown=plot_drawdown,
            plot_trades=plot_trades,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose,
            resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend,
            open_browser=open_browser)


# ============================================================
# __all__ —— 模块公开 API
# ============================================================
# NOTE: Don't put anything public below this __all__ list

__all__ = [getattr(v, '__name__', k)
           for k, v in globals().items()                        # export
           if ((callable(v) and getattr(v, '__module__', None) == __name__ or  # callables from this module
                k.isupper()) and                                # or CONSTANTS
               not getattr(v, '__name__', k).startswith('_'))]  # neither marked internal

# NOTE: Don't put anything public below here. See above.
