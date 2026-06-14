# ============================================================
# backtesting/backtesting.py — 核心回测引擎
# ============================================================
# 上下文层：整个框架的核心。包含 Backtest、Strategy、Order、Trade、
#           Position、内部 _Broker 等全部核心类。
# 功能层：事件驱动逐 K 线回测——Strategy.init() 预计算指标，
#           Strategy.next() 做交易决策，_Broker 管理订单撮合和权益跟踪。
# 设计层：Strategy 用 ABCMeta 元类 + @abstractmethod 强制接口——
#           模板方法模式。Order/Trade/Position 用 @property 提供属性式 API。
#           _Broker 是状态机，_process_orders 处理 stop/limit/market 撮合。
#           optimize 支持网格搜索和 SAMBO 贝叶斯优化，多进程+共享内存。

"""
Core framework data structures.
Objects from this module can also be imported from the top-level
module directly, e.g.

    from backtesting import Backtest, Strategy
"""

from __future__ import annotations

import sys
import warnings
from abc import ABCMeta, abstractmethod
from copy import copy
from difflib import get_close_matches
from functools import lru_cache, partial
from itertools import chain, product, repeat
from math import copysign
from numbers import Number
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng

from ._plotting import plot  # noqa: I001
from ._stats import compute_stats, dummy_stats
from ._util import (
    SharedMemoryManager, _as_str, _Indicator, _Data, _batch,
    _indicator_warmup_nbars, _strategy_indicators, patch, try_, _tqdm,
)

__pdoc__ = {
    'Strategy.__init__': False,
    'Order.__init__': False,
    'Position.__init__': False,
    'Trade.__init__': False,
}


# ═══════════════════════════════════════════════════════════
# Strategy — 策略基类
# ═══════════════════════════════════════════════════════════
# 上下文层：用户必须继承这个类，实现 init() 和 next()。
# 设计层：ABCMeta 元类 + @abstractmethod——强制子类实现接口。
#          模板方法模式：框架调用 init/next，用户填充逻辑。
class Strategy(metaclass=ABCMeta):
    """
    A trading strategy base class. Extend this class and
    override methods Strategy.init and Strategy.next.
    """
    def __init__(self, broker, data, params):
        self._indicators = []         # 已注册的指标列表
        self._broker: _Broker = broker
        self._data: _Data = data
        self._params = self._check_params(params)

    def __repr__(self):
        return '<Strategy ' + str(self) + '>'

    def __str__(self):
        params = ','.join(f'{i[0]}={i[1]}'
                         for i in zip(self._params.keys(),
                                      map(_as_str, self._params.values())))
        if params:
            params = '(' + params + ')'
        return f'{self.__class__.__name__}{params}'

    def _check_params(self, params):
        """校验参数是否在类中有对应类变量定义，用 get_close_matches 给友好提示。"""
        for k, v in params.items():
            if not hasattr(self, k):
                suggestions = get_close_matches(
                    k, (attr for attr in dir(self)
                        if not attr.startswith('_')))
                hint = (f" Did you mean: {', '.join(suggestions)}?"
                        if suggestions else "")
                raise AttributeError(
                    f"Strategy '{self.__class__.__name__}' is missing "
                    f"parameter '{k}'. Strategy class should define "
                    "parameters as class variables before they "
                    "can be optimized or run with." + hint)
            setattr(self, k, v)
        return params

    # ====== I() — 指标注册 ======
    # 用户最频繁调用的 API。init() 里用 self.I() 声明指标，
    # 框架负责：1. 回测中逐步揭示值  2. 图表上绘制指标
    def I(self, func: Callable, *args,
          name=None, plot=True, overlay=None, color=None, scatter=False,
          **kwargs) -> np.ndarray:
        """Declare an indicator. Returns np.ndarray of indicator values."""

        def _format_name(name: str) -> str:
            return name.format(*map(_as_str, args),
                               **dict(zip(kwargs.keys(),
                                         map(_as_str, kwargs.values()))))

        if name is None:
            params = ','.join(filter(None, map(_as_str,
                                               chain(args, kwargs.values()))))
            func_name = _as_str(func)
            name = (f'{func_name}({params})' if params else f'{func_name}')
        elif isinstance(name, str):
            name = _format_name(name)
        elif try_(lambda: all(isinstance(item, str) for item in name), False):
            name = [_format_name(item) for item in name]
        else:
            raise TypeError(
                f'Unexpected `name=` type {type(name)}; '
                'expected `str` or `Sequence[str]`')

        try:
            value = func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f'Indicator "{name}" error. See traceback above.') from e

        if isinstance(value, pd.DataFrame):
            value = value.values.T

        if value is not None:
            value = try_(lambda: np.asarray(value, order='C'), None)
        is_arraylike = bool(value is not None and value.shape)

        if is_arraylike and np.argmax(value.shape) == 0:
            value = value.T

        if isinstance(name, list) and (
                np.atleast_2d(value).shape[0] != len(name)):
            raise ValueError(
                f'Length of `name=` ({len(name)}) must agree with '
                f'the number of arrays the indicator returns '
                f'({value.shape[0]}).')

        if (not is_arraylike or not 1 <= value.ndim <= 2
                or value.shape[-1] != len(self._data.Close)):
            raise ValueError(
                'Indicators must return (optionally a tuple of) numpy.arrays '
                f'of same length as `data` '
                f'(data shape: {self._data.Close.shape}; '
                f'indicator "{name}" shape: {getattr(value, "shape", "")}, '
                f'returned value: {value})')

        # overlay 启发式——指标值大多在 Close 的 60%-140% → 叠加型
        if overlay is None and np.issubdtype(value.dtype, np.number):
            x = value / self._data.Close
            with np.errstate(invalid='ignore'):
                overlay = ((x < 1.4) & (x > .6)).mean() > .6

        value = _Indicator(value, name=name, plot=plot, overlay=overlay,
                           color=color, scatter=scatter,
                           index=self.data.index)
        self._indicators.append(value)
        return value

    @abstractmethod
    def init(self):
        """初始化策略——预计算指标。Override this method."""

    @abstractmethod
    def next(self):
        """每 K 线调用一次——交易决策。Override this method."""

    # 满仓常量——__repr__ 返回 ".9999" 而非 "0.999..."，用户友好
    class __FULL_EQUITY(float):
        def __repr__(self): return '.9999'
    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    def buy(self, *, size: float = _FULL_EQUITY,
            limit: Optional[float] = None, stop: Optional[float] = None,
            sl: Optional[float] = None, tp: Optional[float] = None,
            tag: object = None) -> 'Order':
        """多头进场。size: 0~1=权益比例, >=1=绝对股数。"""
        assert 0 < size < 1 or round(size) == size >= 1, \
            "size must be a positive fraction of equity, or a " \
            "positive whole number of units"
        return self._broker.new_order(size, limit, stop, sl, tp, tag)

    def sell(self, *, size: float = _FULL_EQUITY,
             limit: Optional[float] = None, stop: Optional[float] = None,
             sl: Optional[float] = None, tp: Optional[float] = None,
             tag: object = None) -> 'Order':
        """空头进场。size 为负表示空头方向。"""
        assert 0 < size < 1 or round(size) == size >= 1, \
            "size must be a positive fraction of equity, or a " \
            "positive whole number of units"
        return self._broker.new_order(-size, limit, stop, sl, tp, tag)

    # 属性——用户通过 self.position / self.trades 等访问内部状态
    @property
    def equity(self) -> float:
        return self._broker.equity

    @property
    def data(self) -> _Data:
        """OHLCV 数据。init() 里是全量，next() 里只到当前 bar。"""
        return self._data

    @property
    def position(self) -> 'Position':
        return self._broker.position

    @property
    def orders(self) -> 'Tuple[Order, ...]':
        return tuple(self._broker.orders)

    @property
    def trades(self) -> 'Tuple[Trade, ...]':
        return tuple(self._broker.trades)

    @property
    def closed_trades(self) -> 'Tuple[Trade, ...]':
        return tuple(self._broker.closed_trades)


# ═══════════════════════════════════════════════════════════
# Position — 当前持仓
# ═══════════════════════════════════════════════════════════
class Position:
    """当前持仓。支持 bool(self.position)。"""

    def __init__(self, broker: '_Broker'):
        self.__broker = broker

    def __bool__(self):
        return self.size != 0

    @property
    def size(self) -> float:
        """持仓数量（负=空头）。"""
        return sum(trade.size for trade in self.__broker.trades)

    @property
    def pl(self) -> float:
        """浮动盈亏（现金）。"""
        return sum(trade.pl for trade in self.__broker.trades)

    @property
    def pl_pct(self) -> float:
        """浮动盈亏百分比。"""
        total_invested = sum(
            trade.entry_price * abs(trade.size)
            for trade in self.__broker.trades)
        return (self.pl / total_invested) * 100 if total_invested else 0

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    def close(self, portion: float = 1.):
        """关闭 portion 比例的持仓。"""
        for trade in self.__broker.trades:
            trade.close(portion)

    def __repr__(self):
        return f'<Position: {self.size} ' \
               f'({len(self.__broker.trades)} trades)>'


class _OutOfMoneyError(Exception):
    pass


# ═══════════════════════════════════════════════════════════
# Order — 订单
# ═══════════════════════════════════════════════════════════
# 支持市价单/限价单/止损单/止损限价单，可附带 SL/TP 保护。
class Order:
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
        self.__size = size
        self.__limit_price = limit_price
        self.__stop_price = stop_price
        self.__sl_price = sl_price
        self.__tp_price = tp_price
        self.__parent_trade = parent_trade
        self.__tag = tag

    def _replace(self, **kwargs):
        # name mangling：__size → _Order__size，子类不会覆盖
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def __repr__(self):
        return '<Order {}>'.format(
            ', '.join(f'{param}={try_(lambda: round(value, 5), value)!r}'
                      for param, value in (
                          ('size', self.__size),
                          ('limit', self.__limit_price),
                          ('stop', self.__stop_price),
                          ('sl', self.__sl_price),
                          ('tp', self.__tp_price),
                          ('contingent', self.is_contingent),
                          ('tag', self.__tag),
                      ) if value is not None))

    def cancel(self):
        """取消订单。"""
        self.__broker.orders.remove(self)
        trade = self.__parent_trade
        if trade:
            if self is trade._sl_order:
                trade._replace(sl_order=None)
            elif self is trade._tp_order:
                trade._replace(tp_order=None)

    @property
    def size(self) -> float:
        """订单量（负=空头）。0~1=权益比例，>=1=绝对股数。"""
        return self.__size

    @property
    def limit(self) -> Optional[float]:
        """限价（None=市价单）。"""
        return self.__limit_price

    @property
    def stop(self) -> Optional[float]:
        """止损触发价（触发后转为市价/限价单）。"""
        return self.__stop_price

    @property
    def sl(self) -> Optional[float]:
        """成交后自动挂的止损价。"""
        return self.__sl_price

    @property
    def tp(self) -> Optional[float]:
        """成交后自动挂的止盈价。"""
        return self.__tp_price

    @property
    def parent_trade(self):
        return self.__parent_trade

    @property
    def tag(self):
        """用户自定义标签，会传递到 Trade。"""
        return self.__tag

    __pdoc__['Order.parent_trade'] = False

    @property
    def is_long(self):
        return self.__size > 0

    @property
    def is_short(self):
        return self.__size < 0

    @property
    def is_contingent(self):
        """是否 SL/TP 条件单（OCO——一单二挂）。"""
        return bool((parent := self.__parent_trade) and
                    (self is parent._sl_order or
                     self is parent._tp_order))


# ═══════════════════════════════════════════════════════════
# Trade — 交易（订单成交后产生）
# ═══════════════════════════════════════════════════════════
# 跟踪从进场到出场的全部信息。活跃在 self.trades，已平仓在 self.closed_trades。
class Trade:
    def __init__(self, broker: '_Broker', size: int,
                 entry_price: float, entry_bar, tag):
        self.__broker = broker
        self.__size = size
        self.__entry_price = entry_price
        self.__exit_price: Optional[float] = None
        self.__entry_bar: int = entry_bar
        self.__exit_bar: Optional[int] = None
        self.__sl_order: Optional[Order] = None
        self.__tp_order: Optional[Order] = None
        self.__tag = tag
        self._commissions = 0

    def __repr__(self):
        return (f'<Trade size={self.__size} '
                f'time={self.__entry_bar}-{self.__exit_bar or ""} '
                f'price={self.__entry_price}-{self.__exit_price or ""} '
                f'pl={self.pl:.0f}'
                f'{" tag=" + str(self.__tag) if self.__tag is not None else ""}>')

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def _copy(self, **kwargs):
        return copy(self)._replace(**kwargs)

    def close(self, portion: float = 1.):
        """关闭 portion 比例的交易。"""
        assert 0 < portion <= 1, "portion must be a fraction between 0 and 1"
        size = copysign(
            max(1, int(round(abs(self.__size) * portion))), -self.__size)
        order = Order(self.__broker, size, parent_trade=self, tag=self.__tag)
        self.__broker.orders.insert(0, order)

    @property
    def size(self):
        return self.__size

    @property
    def entry_price(self) -> float:
        return self.__entry_price

    @property
    def exit_price(self) -> Optional[float]:
        return self.__exit_price

    @property
    def entry_bar(self) -> int:
        return self.__entry_bar

    @property
    def exit_bar(self) -> Optional[int]:
        return self.__exit_bar

    @property
    def tag(self):
        return self.__tag

    @property
    def _sl_order(self):
        return self.__sl_order

    @property
    def _tp_order(self):
        return self.__tp_order

    @property
    def entry_time(self) -> Union[pd.Timestamp, int]:
        return self.__broker._data.index[self.__entry_bar]

    @property
    def exit_time(self) -> Optional[Union[pd.Timestamp, int]]:
        if self.__exit_bar is None:
            return None
        return self.__broker._data.index[self.__exit_bar]

    @property
    def is_long(self):
        return self.__size > 0

    @property
    def is_short(self):
        return not self.is_long

    @property
    def pl(self):
        """浮动盈亏（活跃中用当前价，已平仓用出场价）。"""
        price = self.__exit_price or self.__broker.last_price
        return (self.__size * (price - self.__entry_price)) - self._commissions

    @property
    def pl_pct(self):
        price = self.__exit_price or self.__broker.last_price
        gross_pl_pct = copysign(1, self.__size) * (
            price / self.__entry_price - 1)
        commission_pct = self._commissions / (
            abs(self.__size) * self.__entry_price)
        return gross_pl_pct - commission_pct

    @property
    def value(self):
        price = self.__exit_price or self.__broker.last_price
        return abs(self.__size) * price

    # SL/TP 属性——可读写
    @property
    def sl(self):
        return self.__sl_order and self.__sl_order.stop

    @sl.setter
    def sl(self, price: float):
        self.__set_contingent('sl', price)

    @property
    def tp(self):
        return self.__tp_order and self.__tp_order.limit

    @tp.setter
    def tp(self, price: float):
        self.__set_contingent('tp', price)

    def __set_contingent(self, type, price):
        assert type in ('sl', 'tp')
        assert price is None or 0 < price < np.inf, price
        attr = f'_{self.__class__.__qualname__}__{type}_order'
        order: Order = getattr(self, attr)
        if order:
            order.cancel()
        if price:
            kwargs = {'stop': price} if type == 'sl' else {'limit': price}
            order = self.__broker.new_order(
                -self.size, trade=self, tag=self.tag, **kwargs)
            setattr(self, attr, order)


# ═══════════════════════════════════════════════════════════
# _Broker — 内部撮合引擎
# ═══════════════════════════════════════════════════════════
# 上下文层：回测引擎的"心脏"。管理订单撮合、交易状态、保证金、权益。
# 设计层：状态机——每 K 线调用 next() → _process_orders() 处理订单队列。
class _Broker:
    def __init__(self, *, data, cash, spread, commission, margin,
                 trade_on_close, hedging, exclusive_orders, index):
        assert cash > 0 and 0 < margin <= 1
        self._data: _Data = data
        self._cash = cash

        # 佣金支持三种形式：callable / (fixed, relative) / 纯 relative
        if callable(commission):
            self._commission = commission
        else:
            try:
                self._commission_fixed, self._commission_relative = commission
            except TypeError:
                self._commission_fixed, self._commission_relative = 0, commission
            assert self._commission_fixed >= 0
            assert -.1 <= self._commission_relative < .1
            self._commission = self._commission_func

        self._spread = spread
        self._leverage = 1 / margin
        self._trade_on_close = trade_on_close
        self._hedging = hedging
        self._exclusive_orders = exclusive_orders

        self._equity = np.tile(np.nan, len(index))
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.position = Position(self)
        self.closed_trades: List[Trade] = []

    def _commission_func(self, order_size, price):
        return (self._commission_fixed
                + abs(order_size) * price * self._commission_relative)

    def __repr__(self):
        return (f'<Broker: {self._cash:.0f}{self.position.pl:+.1f} '
                f'({len(self.trades)} trades)>')

    def new_order(self, size: float,
                  limit: Optional[float] = None,
                  stop: Optional[float] = None,
                  sl: Optional[float] = None,
                  tp: Optional[float] = None,
                  tag: object = None, *,
                  trade: Optional[Trade] = None) -> Order:
        size = float(size)
        stop = stop and float(stop)
        limit = limit and float(limit)
        sl = sl and float(sl)
        tp = tp and float(tp)

        is_long = size > 0
        assert size != 0
        adjusted_price = self._adjusted_price(size)

        # 验证价格逻辑：多头 SL < 入场价 < TP，空头反过来
        if is_long:
            if not ((sl or -np.inf) < (limit or stop or adjusted_price)
                    < (tp or np.inf)):
                raise ValueError(
                    "Long orders require: "
                    f"SL ({sl}) < LIMIT ({limit or stop or adjusted_price})"
                    f" < TP ({tp})")
        else:
            if not ((tp or -np.inf) < (limit or stop or adjusted_price)
                    < (sl or np.inf)):
                raise ValueError(
                    "Short orders require: "
                    f"TP ({tp}) < LIMIT ({limit or stop or adjusted_price})"
                    f" < SL ({sl})")

        order = Order(self, size, limit, stop, sl, tp, trade, tag)

        # exclusive_orders：新订单自动取消旧挂单、平旧仓位
        if not trade and self._exclusive_orders:
            for o in self.orders:
                if not o.is_contingent:
                    o.cancel()
            for t in self.trades:
                t.close()

        self.orders.insert(
            0 if trade and stop else len(self.orders), order)
        return order

    @property
    def last_price(self) -> float:
        return self._data.Close[-1]

    def _adjusted_price(self, size=None, price=None) -> float:
        """Long 价略高、short 价略低——模拟买卖价差。"""
        return (price or self.last_price) * (
            1 + copysign(self._spread, size))

    @property
    def equity(self) -> float:
        return self._cash + sum(trade.pl for trade in self.trades)

    @property
    def margin_available(self) -> float:
        margin_used = sum(trade.value / self._leverage
                         for trade in self.trades)
        return max(0, self.equity - margin_used)

    def next(self):
        """每 K 线调用一次——处理订单、记录权益、检测爆仓。"""
        i = self._i = len(self._data) - 1
        self._process_orders()

        equity = self.equity
        self._equity[i] = equity

        if equity <= 0:   # 爆仓
            assert self.margin_available <= 0
            for trade in self.trades:
                self._close_trade(trade, self._data.Close[-1], i)
            self._cash = 0
            self._equity[i:] = 0
            raise _OutOfMoneyError

    def _process_orders(self):
        """订单撮合——每 K 线遍历订单队列，按 stop→limit→market 优先级处理。"""
        data = self._data
        open, high, low = data.Open[-1], data.High[-1], data.Low[-1]
        reprocess_orders = False

        for order in list(self.orders):
            if order not in self.orders:
                continue

            # Step 1: 止损触发检测
            stop_price = order.stop
            if stop_price:
                is_stop_hit = (high >= stop_price if order.is_long
                              else low <= stop_price)
                if not is_stop_hit:
                    continue
                order._replace(stop_price=None)

            # Step 2: 限价可达性检测
            if order.limit:
                is_limit_hit = (low <= order.limit if order.is_long
                               else high >= order.limit)
                is_limit_hit_before_stop = (
                    is_limit_hit
                    and (order.limit <= (stop_price or -np.inf)
                         if order.is_long
                         else order.limit >= (stop_price or np.inf)))
                if not is_limit_hit or is_limit_hit_before_stop:
                    continue

                price = (min(stop_price or open, order.limit)
                         if order.is_long
                         else max(stop_price or open, order.limit))
            else:
                prev_close = data.Close[-2]
                price = (prev_close
                         if self._trade_on_close and not order.is_contingent
                         else open)
                if stop_price:
                    price = (max(price, stop_price) if order.is_long
                            else min(price, stop_price))

            # Step 3: 成交时间索引
            is_market_order = not order.limit and not stop_price
            time_index = (
                (self._i - 1)
                if is_market_order and self._trade_on_close
                   and not order.is_contingent
                else self._i)

            # Step 4: contingent 订单（SL/TP）
            if order.parent_trade:
                trade = order.parent_trade
                _prev_size = trade.size
                size = copysign(
                    min(abs(_prev_size), abs(order.size)), order.size)
                if trade in self.trades:
                    self._reduce_trade(trade, price, size, time_index)
                    if price == stop_price:
                        trade._sl_order._replace(stop_price=stop_price)
                if order in (trade._sl_order, trade._tp_order):
                    assert order.size == -trade.size
                    assert order not in self.orders
                else:
                    assert abs(_prev_size) >= abs(size) >= 1
                    self.orders.remove(order)
                continue

            # Step 5: 独立订单
            adjusted_price = self._adjusted_price(order.size, price)
            adjusted_price_plus_commission = (
                adjusted_price
                + self._commission(order.size, price) / abs(order.size))

            size = order.size
            if -1 < size < 1:
                # 比例仓位 → 实际股数
                size = copysign(
                    int((self.margin_available * self._leverage * abs(size))
                        // adjusted_price_plus_commission), size)
                if not size:
                    warnings.warn(
                        f'time={self._i}: Broker canceled the '
                        'relative-sized order due to insufficient margin '
                        f'(equity={self.equity:.2f}, '
                        f'margin_available={self.margin_available:.2f}).',
                        category=UserWarning)
                    self.orders.remove(order)
                    continue
            assert size == round(size)
            need_size = int(size)

            # 非对冲模式：FIFO 平反向仓位
            if not self._hedging:
                for trade in list(self.trades):
                    if trade.is_long == order.is_long:
                        continue
                    if abs(need_size) >= abs(trade.size):
                        self._close_trade(trade, price, time_index)
                        need_size += trade.size
                    else:
                        self._reduce_trade(trade, price, need_size, time_index)
                        need_size = 0
                    if not need_size:
                        break

            # 保证金不足→取消
            if (abs(need_size) * adjusted_price_plus_commission
                    > self.margin_available * self._leverage):
                warnings.warn(
                    f'time={self._i}: Broker canceled the order due to '
                    f'insufficient margin '
                    f'(equity={self.equity:.2f}, '
                    f'margin_available={self.margin_available:.2f}).',
                    category=UserWarning)
                self.orders.remove(order)
                continue

            if need_size:
                self._open_trade(
                    adjusted_price, need_size,
                    order.sl, order.tp, time_index, order.tag)

                # SL/TP 可能在同 K 线触发，需要重新处理
                if order.sl or order.tp:
                    if is_market_order:
                        reprocess_orders = True
                    elif (stop_price and not order.limit and order.tp
                          and ((order.is_long and order.tp <= high
                                and (order.sl or -np.inf) < low)
                               or (order.is_short and order.tp >= low
                                   and (order.sl or np.inf) > high))):
                        reprocess_orders = True
                    elif (low <= (order.sl or -np.inf) <= high
                          or low <= (order.tp or -np.inf) <= high):
                        warnings.warn(
                            f"({data.index[-1]}) A contingent SL/TP order "
                            "would execute in the same bar its parent "
                            "stop/limit order was turned into a trade. "
                            "Since we can't assert the precise intra-candle "
                            "price movement, the affected SL/TP order will "
                            "instead be executed on the next (matching) "
                            "price/bar, making the result (of this trade) "
                            "somewhat dubious.",
                            UserWarning)

            self.orders.remove(order)

        if reprocess_orders:
            self._process_orders()

    def _reduce_trade(self, trade, price, size, time_index):
        """部分平仓。"""
        assert trade.size * size < 0
        assert abs(trade.size) >= abs(size)

        size_left = trade.size + size
        if not size_left:
            close_trade = trade
        else:
            trade._replace(size=size_left)
            if trade._sl_order:
                trade._sl_order._replace(size=-trade.size)
            if trade._tp_order:
                trade._tp_order._replace(size=-trade.size)
            close_trade = trade._copy(
                size=-size, sl_order=None, tp_order=None)
            self.trades.append(close_trade)
        self._close_trade(close_trade, price, time_index)

    def _close_trade(self, trade, price, time_index):
        """完全平仓——移除交易+关联订单，记录佣金。"""
        self.trades.remove(trade)
        if trade._sl_order:
            self.orders.remove(trade._sl_order)
        if trade._tp_order:
            self.orders.remove(trade._tp_order)

        closed_trade = trade._replace(
            exit_price=price, exit_bar=time_index)
        self.closed_trades.append(closed_trade)
        commission = self._commission(trade.size, price)
        self._cash += trade.pl - commission
        trade_open_commission = self._commission(
            closed_trade.size, closed_trade.entry_price)
        closed_trade._commissions = commission + trade_open_commission

    def _open_trade(self, price, size, sl, tp, time_index, tag):
        """开新仓位——创建 Trade + SL/TP 订单。"""
        trade = Trade(self, size, price, time_index, tag)
        self.trades.append(trade)
        self._cash -= self._commission(size, price)
        if tp:
            trade.tp = tp
        if sl:
            trade.sl = sl


# ═══════════════════════════════════════════════════════════
# Backtest — 回测引擎（用户入口）
# ═══════════════════════════════════════════════════════════
# 上下文层：用户直接使用这个类。创建实例→run()执行→plot()可视化。
class Backtest:
    """Backtest a particular strategy on particular data."""

    def __init__(self,
                 data: pd.DataFrame,
                 strategy: Type[Strategy], *,
                 cash: float = 10_000,
                 spread: float = .0,
                 commission: Union[float, Tuple[float, float]] = .0,
                 margin: float = 1.,
                 trade_on_close=False,
                 hedging=False,
                 exclusive_orders=False,
                 finalize_trades=False):
        if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
            raise TypeError('`strategy` must be a Strategy sub-type')
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas.DataFrame with columns")
        if not isinstance(spread, Number):
            raise TypeError('`spread` must be a float')
        if (not isinstance(commission, (Number, tuple))
                and not callable(commission)):
            raise TypeError('`commission` must be float, (fixed, relative), '
                            'or callable')

        data = data.copy(deep=False)

        # 智能日期索引转换
        if (not isinstance(data.index, pd.DatetimeIndex)
                and not isinstance(data.index, pd.RangeIndex)
                and (data.index.is_numeric()
                     and (data.index
                          > pd.Timestamp('1975').timestamp()).mean() > .8)):
            try:
                data.index = pd.to_datetime(
                    data.index, infer_datetime_format=True)
            except ValueError:
                pass

        if 'Volume' not in data:
            data['Volume'] = np.nan

        if len(data) == 0:
            raise ValueError('OHLC `data` is empty')
        if len(data.columns.intersection(
                {'Open', 'High', 'Low', 'Close', 'Volume'})) != 5:
            raise ValueError("`data` must have columns "
                             "Open, High, Low, Close, (Volume)")
        if data[['Open', 'High', 'Low', 'Close']].isnull().values.any():
            raise ValueError('Some OHLC values are missing (NaN).')

        if np.any(data['Close'] > cash):
            warnings.warn(
                'Some prices are larger than initial cash value. '
                'Note that fractional trading is not supported by '
                'this class.')

        if not data.index.is_monotonic_increasing:
            warnings.warn('Data index is not sorted. Sorting.')
            data = data.sort_index()
        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.warn('Data index is not datetime. '
                          'Assuming simple periods.')

        self._data: pd.DataFrame = data
        # partial 延迟 _Broker 创建——每次 run() 重新生成
        self._broker = partial(
            _Broker, cash=cash, spread=spread,
            commission=commission, margin=margin,
            trade_on_close=trade_on_close, hedging=hedging,
            exclusive_orders=exclusive_orders, index=data.index)
        self._strategy = strategy
        self._results: Optional[pd.Series] = None
        self._finalize_trades = bool(finalize_trades)

    # ====== run() — 执行回测 ======
    def run(self, **kwargs) -> pd.Series:
        """
        Run the backtest. Returns pd.Series with results and statistics.
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
            You may obtain different results for different strategy
            parameters. E.g. if you use 50- and 200-bar SMA, the trading
            simulation will begin on bar 201. The actual length of delay
            is equal to the lookback period of the Strategy.I indicator
            which lags the most. This can affect results.
        """
        data = _Data(self._data.copy(deep=False))
        broker: _Broker = self._broker(data=data)
        strategy: Strategy = self._strategy(broker, data, kwargs)

        strategy.init()
        data._update()

        indicator_attrs = _strategy_indicators(strategy)
        # 跳过指标预热期（+1 保证至少两个 bar 可用）
        start = 1 + _indicator_warmup_nbars(strategy)

        with np.errstate(invalid='ignore'):
            # ====== 主循环：逐 K 线推进 ======
            for i in _tqdm(range(start, len(self._data)),
                          desc=self.run.__qualname__,
                          unit='bar', mininterval=2, miniters=100):
                # 逐步揭示数据——只暴露到 i 为止
                data._set_length(i + 1)
                for attr, indicator in indicator_attrs:
                    setattr(strategy, attr, indicator[..., :i + 1])

                try:
                    broker.next()        # 撮合订单
                except _OutOfMoneyError:
                    break

                strategy.next()          # 策略决策
            else:
                if self._finalize_trades is True:
                    for trade in reversed(broker.trades):
                        trade.close()
                    if start < len(self._data):
                        try_(broker.next, exception=_OutOfMoneyError)
                elif len(broker.trades):
                    warnings.warn(
                        'Some trades remain open at the end of backtest. '
                        'Use `Backtest(..., finalize_trades=True)` to close '
                        'them and include them in stats.')

            data._set_length(len(self._data))
            equity = (pd.Series(broker._equity)
                      .bfill().fillna(broker._cash).values)
            self._results = compute_stats(
                trades=broker.closed_trades,
                equity=equity,
                ohlc_data=self._data,
                risk_free_rate=0.0,
                strategy_instance=strategy)

        return self._results

    # ====== optimize() — 参数优化 ======
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
        """优化策略参数。method='grid'→网格搜索，'sambo'→贝叶斯优化。"""

        if not kwargs:
            raise ValueError('Need some strategy parameters to optimize')

        maximize_key = None
        if isinstance(maximize, str):
            maximize_key = str(maximize)
            if maximize not in dummy_stats().index:
                raise ValueError('`maximize` must match a key in '
                                 'backtest.run() result')

            def maximize(stats: pd.Series, _key=maximize):
                return stats[_key]
        elif not callable(maximize):
            raise TypeError('`maximize` must be str or callable')
        assert callable(maximize)

        have_constraint = bool(constraint)
        if constraint is None:
            def constraint(_):
                return True
        elif not callable(constraint):
            raise TypeError('`constraint` must be a function')
        assert callable(constraint)

        if method == 'skopt':
            method = 'sambo'
            warnings.warn('`method="skopt"` is deprecated. '
                          'Use `method="sambo"`.')
        if return_optimization and method != 'sambo':
            raise ValueError(
                "return_optimization=True only valid if method='sambo'")

        def _tuple(x):
            return x if isinstance(x, Sequence) and not isinstance(x, str) \
                   else (x,)

        for k, v in kwargs.items():
            if len(_tuple(v)) == 0:
                raise ValueError(
                    f"Optimization variable '{k}' is passed no values")

        class AttrDict(dict):
            """字典 + 属性访问——constraint 函数里可以 p.fast 而非 p['fast']。"""
            def __getattr__(self, item):
                return self[item]

        def _grid_size():
            size = int(np.prod([len(_tuple(v)) for v in kwargs.values()]))
            if size < 10_000 and have_constraint:
                size = sum(1 for p in product(
                    *(zip(repeat(k), _tuple(v))
                      for k, v in kwargs.items()))
                           if constraint(AttrDict(p)))
            return size

        # --- 网格搜索 ---
        def _optimize_grid():
            rand = default_rng(random_state).random
            grid_frac = (1 if max_tries is None else
                        max_tries if 0 < max_tries <= 1 else
                        max_tries / _grid_size())
            param_combos = [
                dict(params)
                for params in (
                    AttrDict(params)
                    for params in product(
                        *(zip(repeat(k), _tuple(v))
                          for k, v in kwargs.items())))
                if constraint(params) and rand() <= grid_frac]
            if not param_combos:
                raise ValueError(
                    'No admissible parameter combinations to test')

            if len(param_combos) > 300:
                warnings.warn(
                    f'Searching for best of {len(param_combos)} '
                    'configurations.')

            heatmap = pd.Series(
                np.nan, name=maximize_key,
                index=pd.MultiIndex.from_tuples(
                    [p.values() for p in param_combos],
                    names=next(iter(param_combos)).keys()))

            from . import Pool
            with Pool() as pool, SharedMemoryManager() as smm:
                with patch(self, '_data', None):
                    bt = copy(self)
                results = _tqdm(
                    pool.imap(
                        Backtest._mp_task,
                        ((bt, smm.df2shm(self._data), params_batch)
                         for params_batch in _batch(param_combos))),
                    total=len(param_combos), desc='Backtest.optimize')
                for param_batch, result in zip(
                        _batch(param_combos), results):
                    for params, stats in zip(param_batch, result):
                        if stats is not None:
                            heatmap[tuple(params.values())] = maximize(stats)

            if pd.isnull(heatmap).all():
                stats = self.run(**param_combos[0])
            else:
                best_params = heatmap.idxmax(skipna=True)
                stats = self.run(
                    **dict(zip(heatmap.index.names, best_params)))

            if return_heatmap:
                return stats, heatmap
            return stats

        # --- SAMBO 贝叶斯优化 ---
        def _optimize_sambo():
            try:
                import sambo
            except ImportError:
                raise ImportError(
                    "Need package 'sambo' for method='sambo'. "
                    "pip install sambo") from None

            nonlocal max_tries
            max_tries = (200 if max_tries is None else
                        max(1, int(max_tries * _grid_size()))
                        if 0 < max_tries <= 1 else max_tries)

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

            @lru_cache()
            def memoized_run(tup):
                nonlocal maximize, self
                stats = self.run(**dict(tup))
                return -maximize(stats)   # SAMBO 最小化→目标取负

            progress = iter(_tqdm(repeat(None), total=max_tries,
                                  leave=False,
                                  desc=self.optimize.__qualname__,
                                  mininterval=2))
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
                fun=objective_function, bounds=dimensions,
                constraints=cons, max_iter=max_tries,
                method='sceua', rng=random_state)

            stats = self.run(**dict(zip(kwargs.keys(), res.x)))
            output = [stats]

            if return_heatmap:
                heatmap = pd.Series(
                    dict(zip(map(tuple, res.xv), -res.funv)),
                    name=maximize_key)
                heatmap.index.names = kwargs.keys()
                heatmap.sort_index(inplace=True)
                output.append(heatmap)

            if return_optimization:
                output.append(res)

            return stats if len(output) == 1 else tuple(output)

        if method == 'grid':
            output = _optimize_grid()
        elif method in ('sambo', 'skopt'):
            output = _optimize_sambo()
        else:
            raise ValueError(
                f"Method should be 'grid' or 'sambo', not {method!r}")
        return output

    @staticmethod
    def _mp_task(arg):
        """多进程回测任务——从共享内存恢复数据，跑回测，返回统计。"""
        bt, data_shm, params_batch = arg
        bt._data, shm = SharedMemoryManager.shm2df(data_shm)
        try:
            return [stats.filter(regex='^[^_]')
                    if stats['# Trades'] else None
                    for stats in (bt.run(**params)
                                  for params in params_batch)]
        finally:
            for shmem in shm:
                shmem.close()

    def plot(self, *, results: pd.Series = None,
             filename=None, plot_width=None,
             plot_equity=True, plot_return=False, plot_pl=True,
             plot_volume=True, plot_drawdown=False, plot_trades=True,
             smooth_equity=False, relative_equity=True,
             superimpose: Union[bool, str] = True,
             resample=True, reverse_indicators=False,
             show_legend=True, open_browser=True):
        """生成交互式 HTML 图表——委托给 _plotting.plot()。"""
        if results is None:
            if self._results is None:
                raise RuntimeError(
                    'First issue `backtest.run()` to obtain results.')
            results = self._results

        return plot(
            results=results, df=self._data,
            indicators=results._strategy._indicators,
            filename=filename, plot_width=plot_width,
            plot_equity=plot_equity, plot_return=plot_return,
            plot_pl=plot_pl, plot_volume=plot_volume,
            plot_drawdown=plot_drawdown, plot_trades=plot_trades,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose, resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend, open_browser=open_browser)


# ═══════════════════════════════════════════════════════════
# __all__ — 自动生成模块公开 API
# ═══════════════════════════════════════════════════════════

__all__ = [getattr(v, '__name__', k)
           for k, v in globals().items()
           if ((callable(v) and getattr(v, '__module__', None) == __name__
                or k.isupper())
               and not getattr(v, '__name__', k).startswith('_'))]
